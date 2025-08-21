import argparse
import glob
import json
import math
import os
import shutil
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing import event_accumulator

from BASEN import BASEN_OFFLINE
from BASEN_classifier import BASENClassifier
from dataset import load_ClassificationDataset
from util import find_max_epoch, print_size


# ========== validation function ==========
def val(dataloader, net, loss_fn):
    net.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for mixed_audio, eeg, label, _ in dataloader:

            # Load a batch
            mixed_audio = mixed_audio.cuda()
            eeg = eeg.cuda()
            label = label.cuda()

            # Forward pass
            logits = net(mixed_audio, eeg)  # pred: (B, 4)
            loss = loss_fn(logits, label)

            # Record metrics
            val_loss += loss.item() * label.size(0)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

    # Report loss and accuracy
    val_loss /= total
    val_acc = correct / total
    print(f'Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}')
    return val_loss


# ========== training function ==========
def train(num_gpus, rank, exp_path, log, optimization):

    # ========== directories ==========
    log_directory = os.path.join(log["directory"], exp_path, "classification/")
    tb_dir = os.path.join(log_directory, 'tensorboard/')
    latest_ckpt_directory = os.path.join(log_directory, "checkpoint/latest/")
    best_ckpt_directory = os.path.join(log_directory, "checkpoint/best/")
    os.makedirs(tb_dir, exist_ok=True)
    os.makedirs(latest_ckpt_directory, exist_ok=True)
    os.makedirs(best_ckpt_directory, exist_ok=True)

    # ========== load classification data ==========
    trainloader, valloader = load_ClassificationDataset(
        stimulus_wav_dir=trainset_config_all_insutments["stimulus_wav_dir"],
        response_npy_dir=trainset_config_all_insutments["response_npy_dir"],
        batch_size=optimization["batch_size_per_gpu_offline"],
        num_gpus=num_gpus
    )
    print("Classification dataset loaded.")

    # ========== starting point ==========
    n_batchs_train = len(trainloader)

    if log["ckpt_iter"] == 'max':
        ckpt_iter = find_max_epoch(latest_ckpt_directory)  # last iter completed
    else:
        ckpt_iter = log["ckpt_iter"]

    if ckpt_iter != -1:
        cut_off_epoch = math.floor(ckpt_iter / n_batchs_train)  # last epoch completed
        ckpt_iter = cut_off_epoch * n_batchs_train  # last iter completed TRIMMED
    elif ckpt_iter == -1:
        ckpt_iter = 0

    # ========== load tensorboard ==========
    if rank == 0:

        # find the existing tfevents file
        old_tb_files = glob.glob(os.path.join(tb_dir, "events.out.tfevents.*"))

        # create new or load old
        if len(old_tb_files) == 0:
            tb = SummaryWriter(tb_dir)
            last_val_loss = 100.00  # no old logs, nothing to resume from

        elif len(old_tb_files) == 1:
            old_tb_file = old_tb_files[0]
            print(f"Found old TensorBoard file: {old_tb_file}")

            # load old events
            ea = event_accumulator.EventAccumulator(old_tb_file)
            ea.Reload()

            # Try to extract the last "Val/Val-Loss" at ckpt_iter
            last_val_loss = None
            if "Val/Val-Loss" in ea.Tags()['scalars']:
                for event in ea.Scalars("Val/Val-Loss"):
                    if event.step == ckpt_iter:
                        last_val_loss = event.value
                        print("Recovered Val-Loss at iteration {}: {:.4f}".format(ckpt_iter, last_val_loss))
                        break

            # create a new log file (same dir)
            tb = SummaryWriter(tb_dir)

            # re-log all scalars up to ckpt_iter
            for tag in ea.Tags()['scalars']:
                for event in ea.Scalars(tag):
                    if event.step <= ckpt_iter:
                        tb.add_scalar(tag, event.value, event.step)

            tb.flush()
            print(f"Re-logged TensorBoard up to iteration {ckpt_iter}")

            # delete old tfevents file
            os.remove(old_tb_file)
            print(f"Deleted old TensorBoard")

        elif len(old_tb_files) > 1:
            raise RuntimeError(f"Expected exactly 1 tfevents file in {tb_dir}, found {len(old_tb_files)}")

    # ========== define model ==========
    basen = BASEN_OFFLINE(enc_channel=network_config["enc_channel"], feature_channel=network_config["feature_channel"],
                encoder_kernel_size=network_config["encoder_kernel_size"],
                layer_per_stack=network_config["layer_per_stack"], stack=network_config["stacks"],
                CMCA_layer_num=network_config["CMCA_layer_num"]).cuda()
    net = BASENClassifier(basen).cuda()
    print_size(net)

    # ========== define optimizer ==========
    optimizer = torch.optim.Adam(net.parameters(), lr=optimization["learning_rate"])

    # ========== define loss ==========
    criterion = nn.CrossEntropyLoss()

    # ========== load checkpoint ==========
    time0 = time.time()
    if ckpt_iter > 0:
        try:
            # load checkpoint file
            model_path = os.path.join(latest_ckpt_directory, '{}.pkl'.format(ckpt_iter))
            checkpoint = torch.load(model_path, map_location='cpu')

            # feed model dict and optimizer state
            net.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # record training time based on elapsed time
            time0 -= checkpoint['training_time_seconds']
            print('Model at iteration {} has been trained for {} seconds'.format(
                ckpt_iter, checkpoint['training_time_seconds']))
            print('Checkpoint model loaded successfully')
        except:
            ckpt_iter = 0
            print('No valid checkpoint model found, start training from initialization.')
    else:
        ckpt_iter = 0
        print('No valid checkpoint model found, start training from initialization.')

    # ========== training ==========

    cur_iter = ckpt_iter + 1
    epoch = math.floor(cur_iter / n_batchs_train) + 1
    while epoch <= optimization["epochs"]:
        print("\n========== EPOCH ", epoch, " ==========", sep="")

        # for each iteration (a batch)
        for mixed_audio, eeg, label, _ in trainloader:

            # Load a batch
            mixed_audio = mixed_audio.cuda()
            eeg = eeg.cuda()
            label = label.cuda()

            # Forward pass
            logits = net(mixed_audio, eeg)  # pred: (B, 4)

            # Compute loss and optimize
            optimizer.zero_grad()
            loss = criterion(logits, label)

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1e9)
            optimizer.step()

            # save records
            if rank == 0:
                print("\nIteration: {}".format(cur_iter), flush=True)
                print("Training Loss: {:.4f}".format(loss.item()), flush=True)

                val_loss = val(valloader, net, criterion)
                net.train()

                # save to tensorboard
                tb.add_scalar("Train/Train-Loss", loss.item(), cur_iter)
                tb.add_scalar("Train/Gradient-Norm", grad_norm, cur_iter)
                tb.add_scalar("Val/Val-Loss", val_loss, cur_iter)
                tb.flush()

                # save the latest checkpoint
                checkpoint_name = '{}.pkl'.format(cur_iter)
                torch.save({'iter': cur_iter,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'training_time_seconds': int(time.time() - time0)},
                            os.path.join(latest_ckpt_directory, checkpoint_name))

                # delete old checkpoints
                for i in range(cur_iter - n_batchs_train):
                    old_ckpt = os.path.join(latest_ckpt_directory, '{}.pkl'.format(i))
                    try:
                        os.remove(old_ckpt)
                    except:
                        pass

                # save the best checkpoint
                if val_loss < last_val_loss:
                    print('Validation loss decreased from {:.4f} to {:.4f}, saving best checkpoint'.format(last_val_loss, val_loss))
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

            cur_iter += 1

        print('\nEpoch {} done'.format(epoch))
        epoch += 1

    # ========== close tensorboard after training ==========
    if rank == 0:
        tb.close()

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='configs/BASEN.json',
                        help='JSON file for configuration')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process')
    args = parser.parse_args()

    # Parse configs. Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]  # training parameters
    global dist_config
    dist_config = config["dist_config"]  # unused here, only for offline mode
    global network_config
    network_config = config["network_config"]  # to define network
    global trainset_config_all_insutments
    trainset_config_all_insutments = config["trainset_config_all_insutments"]

    num_gpus = torch.cuda.device_count()
    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    train(num_gpus, args.rank, **train_config)
