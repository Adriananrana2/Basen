import argparse
import glob
import json
import math
import numpy as np
import os
import random
import shutil
import time
import torch
import torch.nn as nn
import torchaudio

from BASEN import BASEN_OFFLINE
from dataset import load_CleanNoisyPairDataset
from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
from sisdr_loss import si_sidrloss
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
from util import LinearWarmupCosineDecay
from util import find_max_epoch, print_size

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)


def val(dataloader, net, loss_fn):
    net.eval()
    val_loss = 0
    with torch.no_grad():
        # for each iteration (a batch)
        for mixed_audio, eeg, clean_audio, input_paths in dataloader:

            # Load a batch
            clean_audio = clean_audio.cuda()
            mixed_audio = mixed_audio.cuda()
            eeg = eeg.cuda()

            # === Forward pass ===
            pred = net(mixed_audio, eeg)  # pred: (8, 1, 44100)
            loss = loss_fn(pred, clean_audio).item()
            val_loss += loss

            # Create save directory
            save_dir = os.path.join(os.path.dirname(__file__), "extracted_audio/offline/")

            if network_config["instument_all_0_single_1"] == 0:
                save_dir = os.path.join(save_dir, "all/")
            else:
                save_dir = os.path.join(save_dir, network_config["instument_name"] + "/")
            os.makedirs(save_dir, exist_ok=True)

            # Save each sample using original filename
            for i in range(pred.shape[0]):
                original_path = input_paths[i]  # full path to stimulus_wav
                filename = os.path.basename(original_path)  # e.g., "0001_xyz_stimulus.wav"
                base = "_".join(filename.split("_")[:-1]) + ".wav"  # strip "_stimulus"
                save_path = os.path.join(save_dir, base)
                waveform = pred[i, 0].detach().cpu()
                torchaudio.save(save_path, waveform.unsqueeze(0), sample_rate=44100)

    val_loss /= len(dataloader)
    print(f"Val Avg loss: {val_loss:>8f}")
    return val_loss


def train(num_gpus, rank, group_name, exp_path, log, optimization):

    # ========== directories ==========
    # add offline
    log_directory = os.path.join(log["directory"], exp_path, "offline/")
    # all or single instument
    if network_config["instument_all_0_single_1"] == 0:
        log_directory = os.path.join(log_directory, "all/")
    else:
        log_directory = os.path.join(log_directory, network_config["instument_name"] + "/")
    # tensorboard directory
    tb_dir = os.path.join(log_directory, 'tensorboard/')
    os.makedirs(tb_dir, exist_ok=True)
    # checkpoint latest directory
    latest_ckpt_directory = os.path.join(log_directory, "checkpoint/latest/")
    os.makedirs(latest_ckpt_directory, exist_ok=True)
    # checkpoint best directory
    best_ckpt_directory = os.path.join(log_directory, "checkpoint/best/")
    os.makedirs(best_ckpt_directory, exist_ok=True)

    # ========== load training data ==========
    if network_config["instument_all_0_single_1"] == 0:
        trainloader, valloader = load_CleanNoisyPairDataset(**trainset_config_all_insutments,
                                                            instument_all_0_single_1 = network_config["instument_all_0_single_1"],
                                                            instument_name = network_config["instument_name"],
                                                            batch_size=optimization["batch_size_per_gpu_offline"],
                                                            num_gpus=num_gpus)
    else:
        trainloader, valloader = load_CleanNoisyPairDataset(**trainset_config_single_insutment,
                                                            instument_all_0_single_1 = network_config["instument_all_0_single_1"],
                                                            instument_name = network_config["instument_name"],
                                                            batch_size=optimization["batch_size_per_gpu_offline"],
                                                            num_gpus=num_gpus)
    print('Data loaded')

    # ========== starting point ==========
    n_epochs_train = optimization["epochs"]  # See BASEN.json
    n_batchs_train = len(trainloader)  # Number of batches per epoch in trainloader

    if log["ckpt_iter"] == 'max':
        ckpt_iter = find_max_epoch(latest_ckpt_directory)
    else:
        ckpt_iter = log["ckpt_iter"]
    if ckpt_iter != -1:
        cut_off_epoch = math.floor(ckpt_iter / n_batchs_train)
        ckpt_iter = cut_off_epoch * n_batchs_train - 1

    # ========== load tensorboard ==========
    if rank == 0:

        # find the existing tfevents file
        old_tb_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))

        # creat new or load old
        if len(old_tb_files) == 0:
            tb = SummaryWriter(tb_dir)

        elif len(old_tb_files) == 1:
            old_tb_file = old_tb_files[0]
            print(f"Found old TensorBoard file: {old_tb_file}")
            print(f"Truncating logs at iteration {ckpt_iter}")

            # load old events
            ea = event_accumulator.EventAccumulator(old_tb_file)
            ea.Reload()

            # create a new log file (same dir)
            tb = SummaryWriter(tb_dir)
 
            # copy all events up to ckpt_iter
            for tag in ea.Tags()['scalars']:
                for event in ea.Scalars(tag):
                    if event.step <= ckpt_iter:
                        tb.add_scalar(tag, event.value, event.step)
            tb.flush()
            print(f"Re-logged up to iteration {ckpt_iter}")

            # delete old tfevents file
            os.remove(old_tb_file)
            print(f"Deleted old TensorBoard log: {old_tb_file}")

        elif len(old_tb_files) > 1:
            raise RuntimeError(f"Expected exactly 1 tfevents file in {tb_dir}, found {len(old_tb_files)}")

    # ========== predefine model ==========
    net = BASEN_OFFLINE(enc_channel=network_config["enc_channel"], feature_channel=network_config["feature_channel"],
                encoder_kernel_size=network_config["encoder_kernel_size"],
                layer_per_stack=network_config["layer_per_stack"], stack=network_config["stacks"],
                CMCA_layer_num=network_config["CMCA_layer_num"]).cuda()
    print_size(net)

    # ========== define optimizer ==========
    optimizer = torch.optim.Adam(net.parameters(), lr=optimization["learning_rate"])

    # ========== load checkpoint ==========
    time0 = time.time()
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(latest_ckpt_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration %s has been trained for %s seconds' % (
                ckpt_iter, checkpoint['training_time_seconds']))
            print('checkpoint model loaded successfully')
        except:
            ckpt_iter = -1
            print('No valid checkpoint model found, start training from initialization.')
    else:
        ckpt_iter = -1
        print('No valid checkpoint model found, start training from initialization.')

    # ========== distributed running initialization ==========
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    if rank == 0:
        if not os.path.isdir(latest_ckpt_directory):
            os.makedirs(latest_ckpt_directory)
            os.chmod(latest_ckpt_directory, 0o775)
        print("latest_ckpt_directory: ", latest_ckpt_directory, flush=True, sep="")

    # ========== apply gradient all reduce ==========
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # ========== training ==========
    cur_iter = ckpt_iter + 1

    # define learning rate scheduler
    scheduler = LinearWarmupCosineDecay(
        optimizer,
        lr_max=optimization["learning_rate"],  # Target max learning rate
        n_slide_or_iter=n_epochs_train * n_batchs_train,  # Total number of training steps or epochs
        iteration=cur_iter,  # Current iteration (if resuming training)
        divider=25,  # Initial LR = lr_max / divider
        warmup_proportion=0.05,  # Warmup = first 5% of steps
        phase=('linear', 'cosine'),  # Use linear warmup + cosine decay
    )

    sisdr = si_sidrloss().cuda()
    last_val_loss = 100.0

    epoch = math.floor(cur_iter / n_batchs_train)
    while epoch < optimization["epochs"]:
        print("========== EPOCH ", epoch, " ==========", sep="")

        # for each iteration (a batch)
        for mixed_audio, eeg, clean_audio, _ in trainloader:

            # Load a batch
            clean_audio = clean_audio.cuda()
            mixed_audio = mixed_audio.cuda()
            eeg = eeg.cuda()

            # Forward pass
            pred = net(mixed_audio, eeg)  # pred: (B, 1, 44100)

            # Compute loss and optimize
            optimizer.zero_grad()
            loss = sisdr(pred, clean_audio)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus).item()
            else:
                reduced_loss = loss.item()

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1e9)
            scheduler.step()
            optimizer.step()

            # save records
            if rank == 0:
                print("iteration: {} \ttrain_loss: {:.7f}".format(cur_iter, reduced_loss), flush=True)

                val_loss = val(valloader, net, sisdr)
                net.train()

                # save to tensorboard
                tb.add_scalar("Train/Train-Loss", reduced_loss, cur_iter)
                tb.add_scalar("Train/Gradient-Norm", grad_norm, cur_iter)
                tb.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], cur_iter)
                tb.add_scalar("Val/Val-Loss", val_loss, cur_iter)
                tb.flush()

                # save the latest checkpoint
                checkpoint_name = '{}.pkl'.format(cur_iter)

                torch.save({'iter': cur_iter,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_time_seconds': int(time.time() - time0)},
                            os.path.join(latest_ckpt_directory, checkpoint_name))
                print('latest checkpoint at iteration %s is saved' % cur_iter)

                # delete old checkpoints
                for i in range(cur_iter):
                    old_ckpt = os.path.join(latest_ckpt_directory, '{}.pkl'.format(i))
                    try:
                        os.remove(old_ckpt)
                        print(f"Deleted old checkpoint: {old_ckpt}")
                    except:
                        pass

                # save the best checkpoint
                if val_loss < last_val_loss:
                    print('validation loss decreases from {} to {}, save the best checkpoint'.format(last_val_loss, val_loss))
                    last_val_loss = val_loss
                    checkpoint_name = '{}.pkl'.format(cur_iter)

                    # delete previous best ckpt
                    shutil.rmtree(best_ckpt_directory)  # delete the entire folder
                    os.makedirs(best_ckpt_directory, exist_ok=True)  # recreate empty folder

                    torch.save({'iter': cur_iter,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'training_time_seconds': int(time.time() - time0)},
                               os.path.join(best_ckpt_directory, checkpoint_name))

                    print('best checkpoint at iteration %s is saved' % cur_iter)

            cur_iter += 1

        epoch += 1
        print('epoch {} done'.format(epoch))

    # ========== close tensorboard fter training ==========
    if rank == 0:
        tb.close()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/BASEN.json',
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='xy',
                        help='name of group for distributed')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]  # training parameters
    global dist_config
    dist_config = config["dist_config"]  # to initialize distributed training
    global network_config
    network_config = config["network_config"]  # to define network
    if network_config["instument_all_0_single_1"] == 0:
        global trainset_config_all_insutments
        trainset_config_all_insutments = config["trainset_config_all_insutments"]  # to load all insutments trainset
    else: 
        global trainset_config_single_insutment
        trainset_config_single_insutment = config["trainset_config_single_insutment"]  # to load single insutment trainset

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU. Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, args.group_name, **train_config)
