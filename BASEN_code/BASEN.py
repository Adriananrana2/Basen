import torch
import torch.nn as nn
from torch.autograd import Variable

from utility import models

class EEGEncoder(nn.Module):
    def __init__(self, layer, enc_channel=64, feature_channel=32, kernel_size=32,
                 kernel=3, skip=True, dilated=True):
        super(EEGEncoder, self).__init__()
        # hyper parameters
        self.enc_channel = enc_channel
        self.feature_channel = feature_channel

        self.proj_kernel_size = kernel_size
        self.stride = 8

        self.layer = layer
        self.kernel = kernel
        self.dilated= dilated

        self.projection = nn.Conv1d(128, feature_channel, self.proj_kernel_size, bias=False, stride=self.stride)
        self.encoder = nn.ModuleList([])
        for i in range(layer):
            if self.dilated:
                self.encoder.append(models.DepthConv1d(feature_channel, feature_channel*2, kernel, dilation=2 ** i,
                                                       padding=2 ** i, skip=skip))
            else:
                self.encoder.append(models.DepthConv1d(feature_channel, feature_channel*2, kernel, dilation=1,
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
        #self.num_spk = num_spk

        self.enc_channel = enc_channel
        self.feature_channel = feature_channel
        self.num_spk = 1
        self.encoder_kernel = encoder_kernel_size
        self.stride = 8
        self.win = 32
        self.layer = layer_per_stack
        self.stack = stack
        self.kernel = kernel
        # EEG encoder
        self.spike_encoder = EEGEncoder(layer=8, enc_channel=enc_channel, feature_channel=feature_channel)
        # audio encoder
        self.audio_encoder = nn.Conv1d(1, self.enc_channel, self.encoder_kernel, bias=False, stride=self.stride)
        
        # TCN separation network from Conv-TasNet
        self.TCN = models.TCN(self.enc_channel, self.enc_channel, self.feature_channel, self.feature_channel*4,
                              self.layer, self.stack, self.kernel, CMCA_layer_num=CMCA_layer_num)

        self.receptive_field = self.TCN.receptive_field
        
        # output decoder
        self.decoder = nn.ConvTranspose1d(self.enc_channel, 1, self.encoder_kernel, bias=False, stride=self.stride)

    def pad_signal(self, input):

        # input is the waveforms: (B, T) or (B, 1, T)
        # reshape and padding
        if input.dim() not in [2, 3]:
            raise RuntimeError("Input can only be 2 or 3 dimensional.")
        
        if input.dim() == 2:
            input = input.unsqueeze(1)
        batch_size = input.size(0)
        nsample = input.size(2)
        rest = self.win - (self.stride + nsample % self.win) % self.win
        if rest > 0:
            pad = Variable(torch.zeros(batch_size, 1, rest)).type(input.type())
            input = torch.cat([input, pad], 2)
        
        pad_aux = Variable(torch.zeros(batch_size, 1, self.stride)).type(input.type())
        input = torch.cat([pad_aux, input, pad_aux], 2)

        return input, rest
        
    def forward(self, input, spike_input):
        
        # padding
        output, rest = self.pad_signal(input)
        #spike, rest = self.pad_signal(spike_input)
        batch_size = output.size(0)
        
        # waveform encoder
        enc_output = self.audio_encoder(output)  # B, N, L
        enc_output_spike = self.spike_encoder(spike_input)
        # generate masks
        masks = torch.sigmoid(self.TCN(enc_output, enc_output_spike)).view(batch_size, self.num_spk, self.enc_channel, -1)  # B, C, N, L
        masked_output = enc_output.unsqueeze(1) * masks  # B, C, N, L
        
        # waveform decoder
        output = self.decoder(masked_output.view(batch_size*self.num_spk, self.enc_channel, -1))  # B*C, 1, L
        output = output[:,:,self.stride:-(rest+self.stride)].contiguous()  # B*C, 1, L
        output = output.view(batch_size, self.num_spk, -1)  # B, C, T
        
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
