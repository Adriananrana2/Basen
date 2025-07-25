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

from BASEN import BASEN_ONLINE
from dataset import load_CleanNoisyPairDataset
from distributed import init_distributed, apply_gradient_allreduce
from sisdr_loss import si_sidrloss
from tensorboard.backend.event_processing import event_accumulator
from torch.utils.tensorboard import SummaryWriter
from util import LinearWarmupCosineDecay
from util import find_max_epoch, print_size
from utility.buffer import SlidingAttractorBuffer

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

            # Prepare for sliding window
            total_audio_len = mixed_audio.shape[-1]  # (8, 2, 837900)[-1] = 837900

            audio_window_size = 44100  # 1 second of audio
            audio_window_stride = int(audio_window_size / 5)  # 0.2 seconds = 8820 samples
            audio_ptr = int(audio_window_size * 0.8)  # 35280 (0.8s past + 0.2s new)

            eeg_window_size = 1280  # 1s of EEG at 1280Hz
            eeg_window_stride = int(eeg_window_size / 5)  # 0.2s = 256 samples
            eeg_ptr = int(eeg_window_size * 0.8)  # 1024 (0.8s past + 0.2s new)

            past_extracted_audio = mixed_audio[:, :, :audio_ptr]  # shape: (8, 2, 35280), pseudo attractor at first
            attractor_buffer = SlidingAttractorBuffer(net.audio_encoder, net.instrument_encoder,
                                                      max_len=35308)  # Init sliding buffers for online mode, 0.8 sec of audio + 7 steps
            collected_audio = []  # To save fully extracted audio

            # Sliding window
            while audio_ptr + audio_window_stride <= total_audio_len:
                # === 1. Generate attractor from past extracted audio ===
                # past_extracted_audio shape: (8, 1, 35280)
                # Append 28 zeros for 7 missing steps (kernel continuity)
                padding_audio = torch.zeros(past_extracted_audio.shape[0], past_extracted_audio.shape[1], 28,
                                            device=past_extracted_audio.device)
                padded_past_audio = torch.cat([past_extracted_audio, padding_audio], dim=-1)  # shape: (8, 1, 35308)
                audio_attractor = attractor_buffer.append(padded_past_audio)

                # === 2. Read current 0.2s raw mixture input ===
                current_audio_input = mixed_audio[:, :,
                                      audio_ptr:audio_ptr + audio_window_stride]  # shape: (8, 2, 8820)

                # === 3. Read aligned 1s EEG: [0.8s past + 7 steps of padding + 0.2s now] ===
                past_eeg = eeg[:, :, eeg_ptr - int(eeg_window_size * 0.8): eeg_ptr]  # shape: (8, 20, 1024)
                padding_eeg = torch.zeros(past_eeg.shape[0], past_eeg.shape[1], 7, device=past_eeg.device)
                padded_past_eeg = torch.cat([past_eeg, padding_eeg], dim=-1)  # shape: (8, 20, 1031)
                asec_7stride_eeg = torch.cat(
                    [padded_past_eeg, eeg[:, :, eeg_ptr: eeg_ptr + int(eeg_window_size * 0.2)]],
                    dim=-1)  # shape: (8, 20, 1287)

                # === 4. Forward pass ===
                # current_audio_input ([8, 2, 8820]) 44100*0.2=8820 0.2s audio input
                # asec_7stride_eeg ([8, 20, 1287]) 256*5+7=1287
                # audio_attractor ([8, 64, 8820]) a_s streach to (44100*0.8-32)/4+1 + 7 = 8820
                pred = net(current_audio_input, asec_7stride_eeg, a_s_past=audio_attractor)  # output: (8, 1, 44100)

                # === 5. Collect audio ===
                if len(collected_audio) == 0:
                    collected_audio.append(pred[:, :, :])  # first 1s
                else:
                    collected_audio.append(pred[:, :, -audio_window_stride:])  # last 0.2s (8820 samples)

                # === 6. Update past extracted audio ===
                past_extracted_audio = pred[:, :, -int(audio_window_size * 0.8):]  # shape: (8, 1, 35280)

                # === 7. Advance time pointers by 0.2s
                audio_ptr += audio_window_stride
                eeg_ptr += eeg_window_stride

            # === Save extracted audio ===
            # Concatenate all collected output chunks (1s + 0.2s * 94 = 19s)
            full_extracted = torch.cat(collected_audio, dim=-1)  # shape: (8, 1, 837900)

            # accumulate loss for each 19s prediction
            val_loss += loss_fn(full_extracted, clean_audio).item()

            # Create save directory
            save_dir = os.path.join(os.path.dirname(__file__), "extracted_audio/online/")

            if network_config["instument_all_0_single_1"] == 0:
                save_dir = os.path.join(save_dir, "all/")
            else:
                save_dir = os.path.join(save_dir, network_config["instument_name"] + "/")

            os.makedirs(save_dir, exist_ok=True)

            # Save each sample using original filename
            for i in range(full_extracted.shape[0]):
                original_path = input_paths[i]  # full path to stimulus_wav
                filename = os.path.basename(original_path)  # e.g., "0001_xyz_stimulus.wav"
                base = "_".join(filename.split("_")[:-1]) + ".wav"  # strip "_stimulus"
                save_path = os.path.join(save_dir, base)
                waveform = full_extracted[i, 0].detach().cpu()
                torchaudio.save(save_path, waveform.unsqueeze(0), sample_rate=44100)

    val_loss /= len(dataloader)
    print('Val Avg loss: {:.4f}'.format(val_loss))
    return val_loss


def train(num_gpus, rank, group_name, exp_path, log, optimization):

    # ========== directories ==========
    # add online
    log_directory = os.path.join(log["directory"], exp_path, "online/")
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
                                                            batch_size=optimization["batch_size_per_gpu_online"],
                                                            num_gpus=num_gpus)
    else:
        trainloader, valloader = load_CleanNoisyPairDataset(**trainset_config_single_insutment,
                                                            instument_all_0_single_1 = network_config["instument_all_0_single_1"],
                                                            instument_name = network_config["instument_name"],
                                                            batch_size=optimization["batch_size_per_gpu_online"],
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

        # create new or load old
        if len(old_tb_files) == 0:
            tb = SummaryWriter(tb_dir)
            last_val_loss = 100.00  # no old logs, nothing to resume from

        elif len(old_tb_files) == 1:
            old_tb_file = old_tb_files[0]
            print(f"Found old TensorBoard file: {old_tb_file}")
            print(f"Truncating logs at iteration {ckpt_iter}")

            # load old events
            ea = event_accumulator.EventAccumulator(old_tb_file)
            ea.Reload()

            # Try to extract the last "Val/Val-Loss" at ckpt_iter
            last_val_loss = None
            if "Val/Val-Loss" in ea.Tags()['scalars']:
                for event in ea.Scalars("Val/Val-Loss"):
                    if event.step == ckpt_iter:
                        last_val_loss = event.value
                        print(f"Recovered Val/Val-Loss at step {ckpt_iter}: {last_val_loss}")
                        break

            # create a new log file (same dir)
            tb = SummaryWriter(tb_dir)

            # re-log all scalars up to ckpt_iter
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
    net = BASEN_ONLINE(enc_channel=network_config["enc_channel"], feature_channel=network_config["feature_channel"],
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
    n_slides_train = 91  # ((44100 * 19)-44100) / (44100*0.2) + 1 = 91
    scheduler = LinearWarmupCosineDecay(
        optimizer,
        lr_max=optimization["learning_rate"],  # Target max learning rate
        n_slide_or_iter=n_epochs_train * n_batchs_train * n_slides_train,  # Total number of training steps or epochs
        iteration=cur_iter,  # Current iteration (if resuming training)
        divider=25,  # Initial LR = lr_max / divider
        warmup_proportion=0.05,  # Warmup = first 5% of steps
        phase=('linear', 'cosine'),  # Use linear warmup + cosine decay
    )

    sisdr = si_sidrloss().cuda()

    epoch = math.floor(cur_iter / n_batchs_train)
    while epoch < optimization["epochs"]:
        print("========== EPOCH ", epoch, " ==========", sep="")

        # for each iteration (a batch)
        for mixed_audio, eeg, clean_audio, _ in trainloader:

            # Load a batch
            clean_audio = clean_audio.cuda()
            mixed_audio = mixed_audio.cuda()
            eeg = eeg.cuda()

            # Prepare for sliding window
            total_audio_len = mixed_audio.shape[-1]  # (8, 2, 837900)[-1] = 837900

            audio_window_size = 44100  # 1 second of audio
            audio_window_stride = int(audio_window_size / 5)  # 0.2 seconds = 8820 samples
            audio_ptr = int(audio_window_size * 0.8)  # 35280 (0.8s past + 0.2s new)

            eeg_window_size = 1280  # 1s of EEG at 1280Hz
            eeg_window_stride = int(eeg_window_size / 5)  # 0.2s = 256 samples
            eeg_ptr = int(eeg_window_size * 0.8)  # 1024 (0.8s past + 0.2s new)

            past_extracted_audio = mixed_audio[:, :, :audio_ptr]  # shape: (8, 2, 35280), pseudo attractor at first
            attractor_buffer = SlidingAttractorBuffer(net.audio_encoder, net.instrument_encoder,
                                                      max_len=35308)  # Init sliding buffers for online mode, 0.8 sec of audio + 7 steps
            collected_audio = []  # To save fully extracted audio
            # Sliding window
            n_slide = 0
            while audio_ptr + audio_window_stride <= total_audio_len:
                n_slide += 1
                # === 1. Generate attractor from past extracted audio ===
                # past_extracted_audio shape: (8, 1, 35280)
                # Append 28 zeros for 7 missing steps (kernel continuity)
                audio_attractor = attractor_buffer.append(past_extracted_audio)

                # === 2. Read current 0.2s raw mixture input ===
                current_audio_input = mixed_audio[:, :,
                                      audio_ptr:audio_ptr + audio_window_stride]  # shape: (8, 2, 8820)

                # === 3. Read aligned 1s EEG: [0.8s past + 7 steps of padding + 0.2s now] ===
                past_eeg = eeg[:, :, eeg_ptr - int(eeg_window_size * 0.8): eeg_ptr]  # shape: (8, 20, 1024)
                padding_eeg = torch.zeros(past_eeg.shape[0], past_eeg.shape[1], 7, device=past_eeg.device)
                padded_past_eeg = torch.cat([past_eeg, padding_eeg], dim=-1)  # shape: (8, 20, 1031)
                asec_7stride_eeg = torch.cat(
                    [padded_past_eeg, eeg[:, :, eeg_ptr: eeg_ptr + int(eeg_window_size * 0.2)]],
                    dim=-1)  # shape: (8, 20, 1287)

                # === 4. Read corresponding 1s clean target ===
                current_clean = clean_audio[:, :, audio_ptr - int(audio_window_size * 0.8): audio_ptr + int(
                    audio_window_size * 0.2)]  # shape: (8, 1, 44100)

                # === 5. Forward pass ===
                # current_audio_input ([8, 2, 8820]) 44100*0.2=8820 0.2s audio input
                # asec_7stride_eeg ([8, 20, 1287]) 256*5+7=1287
                # audio_attractor ([8, 64, 8820]) a_s streach to (44100*0.8-32)/4+1 + 7 = 8820
                pred = net(current_audio_input, asec_7stride_eeg, a_s_past=audio_attractor)  # output: (8, 1, 44100)

                # === 6. Collect audio ===
                if len(collected_audio) == 0:
                    collected_audio.append(pred[:, :, :])  # first 1s
                else:
                    collected_audio.append(pred[:, :, -audio_window_stride:])  # last 0.2s (8820 samples)

                # === 7. Compute loss and optimize ===
                optimizer.zero_grad()
                loss = sisdr(pred, current_clean)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1e9)
                scheduler.step()
                optimizer.step()

                # === 8. Update past extracted audio ===
                past_extracted_audio = pred[:, :, -int(audio_window_size * 0.8):]  # shape: (8, 1, 35280)

                # Advance time pointers by 0.2s
                audio_ptr += audio_window_stride
                eeg_ptr += eeg_window_stride

            # Concatenate all collected output chunks (1s + 0.2s * 94 = 19s)
            full_extracted = torch.cat(collected_audio, dim=-1)  # shape: (8, 1, 837900)

            # calculate train loss
            train_loss = sisdr(full_extracted, clean_audio)

            # save records
            if rank == 0:
                print("iteration: {} \ttrain_loss: {:.4f}".format(cur_iter, train_loss), flush=True)

                val_loss = val(valloader, net, sisdr)
                net.train()

                # save to tensorboard
                tb.add_scalar("Train/Train-Loss", train_loss, cur_iter)
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
                for i in range(cur_iter - n_batchs_train):
                    old_ckpt = os.path.join(latest_ckpt_directory, '{}.pkl'.format(i))
                    try:
                        os.remove(old_ckpt)
                    except:
                        pass

                # save the best checkpoint
                if val_loss < last_val_loss:
                    print('validation loss decreases from {:.4f} to {:.4f}, save the best checkpoint'.format(last_val_loss, val_loss))
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
