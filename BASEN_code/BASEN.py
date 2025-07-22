import torch
import torch.nn as nn
from torch.autograd import Variable

from utility import models


class EEGEncoder(nn.Module):
    def __init__(self, layer, enc_channel=64, feature_channel=32, proj_kernel_size=1,
                 kernel_size=3, skip=True, dilated=False):
        super(EEGEncoder, self).__init__()
        # hyper parameters
        self.enc_channel = enc_channel
        self.feature_channel = feature_channel

        self.proj_kernel_size = proj_kernel_size
        self.stride = 1

        self.layer = layer
        self.kernel_size = kernel_size
        self.dilated = dilated

        self.projection = nn.Conv1d(20, feature_channel, self.proj_kernel_size, bias=False, stride=self.stride)
        self.encoder = nn.ModuleList([])
        for i in range(layer):
            if self.dilated:
                self.encoder.append(
                    models.DepthConv1d(feature_channel, feature_channel * 2, kernel_size, dilation=2 ** i,
                                       padding=2 ** i, skip=skip))
            else:
                self.encoder.append(models.DepthConv1d(feature_channel, feature_channel * 2, kernel_size, dilation=1,
                                                       padding=1, skip=skip))
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(feature_channel, feature_channel, 1)
                                    )

    def forward(self, spike):
        output = self.projection(spike)
        for i in range(len(self.encoder)):
            residual, skip = self.encoder[i](output)
            output = output + residual

        return output


class InstrumentEncoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, n_res_blocks=3):
        super().__init__()
        self.proj = nn.Conv1d(input_channels, 128, kernel_size=1)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.GroupNorm(1, 128),
                nn.PReLU(),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.GroupNorm(1, 128)
            ) for _ in range(n_res_blocks)
        ])
        self.out_prelu = nn.PReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(32)
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_dim, batch_first=True)

    def forward(self, x):  # x: [B, C, T]
        x = self.proj(x)
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = x + residual
        x = self.out_prelu(x)
        x = self.maxpool(x)
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        return h_n.squeeze(0)  # [B, hidden_dim]


class BASEN(nn.Module):
    """
        The Brain-Assisted Speech Enhancement Network, This is adapted from Conv-TasNet.

        args:
        enc_channel: The encoder channel num
        feature_channel: The hidden channel num in separation network
        encoder_kernel_size: Kernel size of the audio encoder and eeg encoder
        layer_per_stack: layer num of every stack
        stack: The num of stacks
        kernel: Kernel size in separation network
    """

    def __init__(self, enc_channel=64, feature_channel=32, encoder_kernel_size=32, layer_per_stack=8, stack=3,
                 kernel=3, CMCA_layer_num=3):
        super(BASEN, self).__init__()

        # hyper parameters
        self.enc_channel = enc_channel
        self.feature_channel = feature_channel
        self.num_spk = 1
        self.encoder_kernel_size = encoder_kernel_size
        self.stride = 4
        self.win = 32
        self.layer = layer_per_stack
        self.stack = stack
        self.kernel = kernel

        # EEG encoder
        self.attractor_dim = 64
        self.instrument_encoder = InstrumentEncoder(input_channels=self.enc_channel, hidden_dim=self.attractor_dim)
        self.spike_encoder = EEGEncoder(layer=8, enc_channel=enc_channel, feature_channel=feature_channel)

        # audio encoder
        self.audio_encoder = nn.Conv1d(2, self.enc_channel, self.encoder_kernel_size, bias=False, stride=self.stride)

        # TCN separation network from Conv-TasNet
        self.TCN = models.TCN(self.enc_channel, self.enc_channel, self.feature_channel, self.feature_channel * 4,
                              self.layer, self.stack, self.kernel, CMCA_layer_num=CMCA_layer_num)

        self.receptive_field = self.TCN.receptive_field

        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_channel, 1, self.encoder_kernel_size, bias=False, stride=self.stride)

    def forward(self, input, spike_input, a_s_past):
        batch_size = input.size(0)

        # === 1. Encode raw audio ===
        # input shape: (8, 2, 8820)  # raw audio for current 0.2s
        # --> encoded shape: (8, 64, 2198)  # (8820 - 32)//4 + 1 = 2198
        enc_output = self.audio_encoder(input)

        # === 2. Encode EEG ===
        # spike_input shape: (8, 20, 1280)  # EEG for 1s (0.8s past + 0.2s now)
        # --> encoded EEG: (8, 32, 1280)  # kernel=1, stride=1 → same length
        spike_input = torch.nn.functional.interpolate(
            spike_input,
            size=11018,  # desired time dimension length
            mode='linear',  # linear interpolation for time series
            align_corners=False)  # Result shape: (8, 20, 11018)

        enc_output_spike = self.spike_encoder(spike_input)

        # === 3. Concatenate attractor with current encoded audio ===
        # a_s_past: (8, 64, 8813)  # past 0.8s encoded
        # enc_output: (8, 64, 2198)  # current 0.2s encoded
        # --> a_s_full: (8, 64, 11011)  # combined 1s sliding window
        a_s_full = torch.cat([a_s_past, enc_output], dim=-1)
        a_s_full = torch.nn.functional.interpolate(
            a_s_full,
            size=11018,  # desired time dimension length
            mode='linear',  # linear interpolation for time series
            align_corners=False)  # Result shape: (8, 20, 11018)

        # === 4. Apply TCN and masks ===
        masks = torch.sigmoid(self.TCN(a_s_full, enc_output_spike)).view(
            batch_size, self.num_spk, self.enc_channel, -1
        )
        # masks shape: (8, 1, 64, 11011)
        # Apply masks to full 1s sliding window
        masked_output = a_s_full.unsqueeze(1) * masks  # (8, 1, 64, 11011)

        # === 5. Decode waveform ===
        # masked_output reshaped: (8, 64, 11011)
        # output shape after decoder: (8, 1, 44132)
        output = self.decoder(masked_output.view(batch_size * self.num_spk, self.enc_channel, -1))

        # Final output shape: (8, 1, 44100)  # clean 1s audio
        output = output.view(batch_size, self.num_spk, -1)

        return output


def test_conv_tasnet():
    x = torch.rand(8, 1, 29180)
    y = torch.rand(8, 128, 29180)
    nnet = BASEN()
    x = nnet(x, y)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()


class BASEN_OFFLINE(nn.Module):
    """
        The Brain-Assisted Speech Enhancement Network, This is adapted from Conv-TasNet.

        args:
        enc_channel: The encoder channel num
        feature_channel: The hidden channel num in separation network
        encoder_kernel_size: Kernel size of the audio encoder and eeg encoder
        layer_per_stack: layer num of every stack
        stack: The num of stacks
        kernel: Kernel size in separation network
    """

    def __init__(self, enc_channel=64, feature_channel=32, encoder_kernel_size=32, layer_per_stack=8, stack=3,
                 kernel=3, CMCA_layer_num=3):
        super(BASEN_OFFLINE, self).__init__()

        # hyper parameters
        self.enc_channel = enc_channel
        self.feature_channel = feature_channel
        self.num_spk = 1
        self.encoder_kernel_size = encoder_kernel_size
        self.stride = 4
        self.win = 32
        self.layer = layer_per_stack
        self.stack = stack
        self.kernel = kernel

        # EEG encoder
        self.attractor_dim = 64
        self.instrument_encoder = InstrumentEncoder(input_channels=self.enc_channel, hidden_dim=self.attractor_dim)
        self.spike_encoder = EEGEncoder(layer=8, enc_channel=enc_channel, feature_channel=feature_channel)

        # audio encoder
        self.audio_encoder = nn.Conv1d(2, self.enc_channel, self.encoder_kernel_size, bias=False, stride=self.stride)

        # TCN separation network from Conv-TasNet
        self.TCN = models.TCN(self.enc_channel, self.enc_channel, self.feature_channel, self.feature_channel * 4,
                              self.layer, self.stack, self.kernel, CMCA_layer_num=CMCA_layer_num)

        self.receptive_field = self.TCN.receptive_field

        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_channel, 1, self.encoder_kernel_size, bias=False, stride=self.stride)

    def forward(self, input, spike_input):
        batch_size = input.size(0)

        # === 1. Encode raw audio ===
        # input shape: (8, 2, 8820)  # raw audio for current 0.2s
        # --> encoded shape: (8, 64, 2198)  # (8820 - 32)//4 + 1 = 2198
        enc_output = self.audio_encoder(input)

        # === 2. Encode EEG ===
        spike_input = torch.nn.functional.interpolate(
            spike_input,
            size=209468,  # desired time dimension length
            mode='linear',  # linear interpolation for time series
            align_corners=False)  # Result shape: (8, 20, 11018)

        enc_output_spike = self.spike_encoder(spike_input)

        # === 4. Apply TCN and masks ===
        masks = torch.sigmoid(self.TCN(enc_output, enc_output_spike)).view(
            batch_size, self.num_spk, self.enc_channel, -1
        )
        # masks shape: (8, 1, 64, 11011)
        masked_output = enc_output.unsqueeze(1) * masks  # (8, 1, 64, 11011)

        # === 5. Decode waveform ===
        output = self.decoder(masked_output.view(batch_size * self.num_spk, self.enc_channel, -1))

        # Final output shape: (8, 1, 44100)  # clean 1s audio
        output = output.view(batch_size, self.num_spk, -1)

        return output


def test_conv_tasnet():
    x = torch.rand(8, 1, 29180)
    y = torch.rand(8, 128, 29180)
    nnet = BASEN()
    x = nnet(x, y)
    s1 = x[0]
    print(s1.shape)


if __name__ == "__main__":
    test_conv_tasnet()
