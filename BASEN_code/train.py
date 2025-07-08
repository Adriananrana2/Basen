from utility.buffer import SlidingAttractorBuffer
import os
import time
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.tensorboard import SummaryWriter

import random

random.seed(0)
torch.manual_seed(0)
np.random.seed(0)

from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor

from dataset import load_CleanNoisyPairDataset
from sisdr_loss import si_sidrloss
from util import rescale, find_max_epoch, print_size
from util import LinearWarmupCosineDecay, loss_fn

from BASEN import BASEN
from torchinfo import summary


def val(dataloader, net, loss_fn):
    net.eval()
    test_loss, correct = 0, 0
    total_iter = 0
    with torch.no_grad():
        # for each iteration (a batch)
        for mixed_audio, eeg, clean_audio, _ in dataloader:

            # Load a batch
            clean_audio = clean_audio.cuda()
            mixed_audio = mixed_audio.cuda()
            eeg = eeg.cuda()

            # Prepare for sliding window
            total_audio_len = mixed_audio.shape[-1]  # (8, 2, 837900)[-1] = 837900
            audio_window_size = 44100  # 1 second of audio
            audio_window_stride = int(audio_window_size / 5)  # 0.2 seconds = 8820 samples
            eeg_window_size = 1280  # 1s of EEG at 1280Hz
            eeg_window_stride = int(eeg_window_size / 5)  # 0.2s = 256 samples
            audio_ptr = int(audio_window_size * 0.8)  # 35280 (0.8s past + 0.2s new)
            eeg_ptr = int(eeg_window_size * 0.8)  # 1024 (0.8s past + 0.2s new)
            past_extracted_audio = mixed_audio[:, :, :audio_ptr]  # shape: (8, 2, 35280), pseudo attractor at first
            attractor_buffer = SlidingAttractorBuffer(net.audio_encoder, net.instrument_encoder,
                                                      max_len=35308)  # Init sliding buffers for online mode, 0.8 sec of audio + 7 steps

            # Sliding window
            while audio_ptr + audio_window_stride <= total_audio_len:
                total_iter += 1
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

                # === 4. Read corresponding 1s clean target ===
                current_clean = clean_audio[:, :, audio_ptr - int(audio_window_size * 0.8): audio_ptr + int(
                    audio_window_size * 0.2)]  # shape: (8, 1, 44100)

                # === 5. Forward pass ===
                # current_audio_input ([8, 2, 8820]) 44100*0.2=8820 0.2s audio input
                # asec_7stride_eeg ([8, 20, 1287]) 256*5+7=1287
                # audio_attractor ([8, 64, 8820]) a_s streach to (44100*0.8-32)/4+1 + 7 = 8820
                pred = net(current_audio_input, asec_7stride_eeg, a_s_past=audio_attractor)  # output: (8, 1, 44100)
                test_loss += loss_fn(pred, current_clean).item()

                # === 6. Update past extracted audio ===
                past_extracted_audio = pred[:, :, -int(audio_window_size * 0.8):]  # shape: (8, 1, 35280)

                # === 7. Advance time pointers by 0.2s
                total_iter += 1
                audio_ptr += audio_window_stride
                eeg_ptr += eeg_window_stride

    test_loss /= total_iter
    print(f"Val Avg loss: {test_loss:>8f}")
    return test_loss


def train(num_gpus, rank, group_name,
          exp_path, log, optimization):
    # setup local experiment path
    if rank == 0:
        print('exp_path:', exp_path, sep="")

    # Create tensorboard logger.
    log_directory = os.path.join(log["directory"], exp_path)
    if rank == 0:
        tb = SummaryWriter(os.path.join(log_directory, 'tensorboard'))

    # distributed running initialization
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    # Get shared ckpt_directory ready
    ckpt_directory = os.path.join(log_directory, 'checkpoint')
    if rank == 0:
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
            os.chmod(ckpt_directory, 0o775)
        print("ckpt_directory: ", ckpt_directory, flush=True, sep="")

    # load training data
    trainloader, valloader = load_CleanNoisyPairDataset(**trainset_config,
                                                        batch_size=optimization["batch_size_per_gpu"],
                                                        num_gpus=num_gpus)
    print('Data loaded')

    # predefine model
    net = BASEN(enc_channel=network_config["enc_channel"], feature_channel=network_config["feature_channel"],
                encoder_kernel_size=network_config["encoder_kernel_size"],
                layer_per_stack=network_config["layer_per_stack"], stack=network_config["stacks"],
                CMCA_layer_num=network_config["CMCA_layer_num"]).cuda()
    # summary(net, input_size=[(8, 1, 29184), (8, 128, 29184)])
    print_size(net)

    # apply gradient all reduce
    if num_gpus > 1:
        net = apply_gradient_allreduce(net)

    # define optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=optimization["learning_rate"])

    # load checkpoint
    time0 = time.time()
    if log["ckpt_iter"] == 'max':
        ckpt_iter = find_max_epoch(ckpt_directory)
    else:
        ckpt_iter = log["ckpt_iter"]
    if ckpt_iter >= 0:
        try:
            # load checkpoint file
            model_path = os.path.join(ckpt_directory, '{}.pkl'.format(ckpt_iter))
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

    # training
    n_iter = ckpt_iter + 1

    # define learning rate scheduler
    scheduler = LinearWarmupCosineDecay(
        optimizer,
        lr_max=optimization["learning_rate"],
        n_iter=optimization["epochs"],
        iteration=n_iter,
        divider=25,
        warmup_proportion=0.05,
        phase=('linear', 'cosine'),
    )

    sisdr = si_sidrloss().cuda()

    last_val_loss = 100.0
    last_sample_loss = 100.0
    epoch = 0
    # data set / batch sizt = iterations for 1 epoch
    while epoch < optimization["epochs"]:
        print("========== EPOCH ", epoch, " ==========", sep="")
        # for each iteration (a batch)
        for mixed_audio, eeg, clean_audio, input_paths in trainloader:

            # Load a batch
            clean_audio = clean_audio.cuda()
            mixed_audio = mixed_audio.cuda()
            eeg = eeg.cuda()

            # Prepare for sliding window
            total_audio_len = mixed_audio.shape[-1]  # (8, 2, 837900)[-1] = 837900
            audio_window_size = 44100  # 1 second of audio
            audio_window_stride = int(audio_window_size / 5)  # 0.2 seconds = 8820 samples
            eeg_window_size = 1280  # 1s of EEG at 1280Hz
            eeg_window_stride = int(eeg_window_size / 5)  # 0.2s = 256 samples
            audio_ptr = int(audio_window_size * 0.8)  # 35280 (0.8s past + 0.2s new)
            eeg_ptr = int(eeg_window_size * 0.8)  # 1024 (0.8s past + 0.2s new)
            past_extracted_audio = mixed_audio[:, :, :audio_ptr]  # shape: (8, 2, 35280), pseudo attractor at first
            collected_audio = []  # To save fully extracted audio
            attractor_buffer = SlidingAttractorBuffer(net.audio_encoder, net.instrument_encoder,
                                                      max_len=35308)  # Init sliding buffers for online mode, 0.8 sec of audio + 7 steps

            # Sliding window
            n_slide = 0
            cur_sample_loss = 0
            while audio_ptr + audio_window_stride <= total_audio_len:
                n_slide += 1
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

                # === 4. Read corresponding 1s clean target ===
                current_clean = clean_audio[:, :, audio_ptr - int(audio_window_size * 0.8): audio_ptr + int(
                    audio_window_size * 0.2)]  # shape: (8, 1, 44100)

                # === 5. Forward pass ===
                # current_audio_input ([8, 2, 8820]) 44100*0.2=8820 0.2s audio input
                # asec_7stride_eeg ([8, 20, 1287]) 256*5+7=1287
                # audio_attractor ([8, 64, 8820]) a_s streach to (44100*0.8-32)/4+1 + 7 = 8820
                output = net(current_audio_input, asec_7stride_eeg, a_s_past=audio_attractor)  # output: (8, 1, 44100)

                # === 6. Collect audio ===
                if len(collected_audio) == 0:
                    collected_audio.append(output[:, :, :])  # first 1s
                else:
                    collected_audio.append(output[:, :, -audio_window_stride:])  # last 0.2s (8820 samples)

                # === 7. Compute loss and optimize ===
                optimizer.zero_grad()
                loss = sisdr(output, current_clean)
                if num_gpus > 1:
                    reduced_loss = reduce_tensor(loss.data, num_gpus).item()
                else:
                    reduced_loss = loss.item()
                cur_sample_loss += reduced_loss
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(net.parameters(), 1e9)
                scheduler.step()
                optimizer.step()

                # === 8. Update past extracted audio ===
                past_extracted_audio = output[:, :, -int(audio_window_size * 0.8):]  # shape: (8, 1, 35280)

                # Advance time pointers by 0.2s
                audio_ptr += audio_window_stride
                eeg_ptr += eeg_window_stride

            n_iter += 1
            cur_sample_loss /= n_slide
            # save checkpoint
            if n_iter > 0 and n_iter % log["iters_per_ckpt"] == 0 and rank == 0:
                print("iteration: {} \tsample loss: {:.7f}".format(n_iter, reduced_loss), flush=True)

                val_loss = val(valloader, net, sisdr)
                net.train()

                if rank == 0:
                    # save to tensorboard
                    tb.add_scalar("Train/Train-Loss", loss.item(), n_iter)
                    tb.add_scalar("Train/Train-Reduced-Loss", reduced_loss, n_iter)
                    tb.add_scalar("Train/Gradient-Norm", grad_norm, n_iter)
                    tb.add_scalar("Train/learning-rate", optimizer.param_groups[0]["lr"], n_iter)
                    tb.add_scalar("Val/Val-Loss", val_loss, n_iter)
                if val_loss < last_val_loss:
                    print(
                        'validation loss decreases from {} to {}, save checkpoint'.format(last_val_loss, val_loss))
                    last_val_loss = val_loss
                    checkpoint_name = '{}.pkl'.format(n_iter)
                    torch.save({'iter': n_iter,
                                'model_state_dict': net.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'training_time_seconds': int(time.time() - time0)},
                               os.path.join(ckpt_directory, checkpoint_name))
                    print('model at iteration %s is saved' % n_iter)
                else:
                    print('validation loss did not decrease')

            # === Save extracted audio ===
            if last_sample_loss > cur_sample_loss:
                print("Sample loss dropped from ", last_sample_loss, " to ", cur_sample_loss, ". Start saving audio",
                      sep="")
                last_sample_loss = cur_sample_loss
                # Concatenate all collected output chunks (1s + 0.2s * 94 = 19s)
                full_extracted = torch.cat(collected_audio, dim=-1)  # shape: (8, 1, 837900)

                # Create save directory
                save_dir = os.path.join(os.path.dirname(__file__), "extracted_audio")
                os.makedirs(save_dir, exist_ok=True)

                # Save each sample using original filename
                for i in range(full_extracted.shape[0]):
                    original_path = input_paths[i]  # full path to stimulus_wav
                    filename = os.path.basename(original_path)  # e.g., "0001_xyz_stimulus.wav"
                    base = "_".join(filename.split("_")[:-1]) + ".wav"  # strip "_stimulus"
                    save_path = os.path.join(save_dir, base)

                    waveform = full_extracted[i, 0].detach().cpu()
                    torchaudio.save(save_path, waveform.unsqueeze(0), sample_rate=44100)

        epoch += 1
        print('epoch {} done'.format(epoch))
    # After training, close TensorBoard.
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
    global trainset_config
    trainset_config = config["trainset_config"]  # to load trainset

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
