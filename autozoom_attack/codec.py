import os

import torch
from torch import nn
import glog as log


class Codec(nn.Module):
    def __init__(self, img_size, in_channels, compress_mode=1, resize=None, use_tanh=True):
        super(Codec, self).__init__()
        self.compress_mode = compress_mode
        self.use_tanh = use_tanh
        working_img_size = img_size
        if resize:
            working_img_size = resize
        self.encoder = nn.ModuleList()
        self.encoder.input_shape = (img_size, img_size)
        if resize:
            self.encoder.append(nn.UpsamplingBilinear2d(size=(resize, resize)))
        self.encoder.append(nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=3, stride=1, padding=1)) # cifar=32 (imagenet=299)
        self.encoder.append(nn.BatchNorm2d(16))
        if use_tanh:
            self.encoder.append(nn.Tanh())
        else:
            self.encoder.append(nn.ReLU())
        self.encoder.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))  # 16      (imagenet=149)
        working_img_size //= 2
        if compress_mode >=2:
            self.encoder.append(nn.Conv2d(16, 16, 3, 1, padding=1))  # 16
            self.encoder.append(nn.BatchNorm2d(16))
            if use_tanh:
                self.encoder.append(nn.Tanh())
            else:
                self.encoder.append(nn.ReLU())
            self.encoder.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))  # 8  (imagenet=74)
            working_img_size //= 2
        if compress_mode >= 3:  # ImageNet
            self.encoder.append(nn.Conv2d(16, 16, 3, 1, padding=1))
            self.encoder.append(nn.BatchNorm2d(16))
            if use_tanh:
                self.encoder.append(nn.Tanh())
            else:
                self.encoder.append(nn.ReLU())
            self.encoder.append(nn.MaxPool2d(kernel_size=2, stride=2, padding=0))  # imagenet=37
            working_img_size //= 2
        self.encoder.append(nn.Conv2d(16, in_channels, 3, 1, padding=1))  # 8
        self.encoder.append(nn.BatchNorm2d(in_channels))
        self.encoder.output_shape = (working_img_size, working_img_size)
        self.decoder = nn.ModuleList()
        self.decoder.input_shape = (working_img_size, working_img_size)
        out_channels = 3
        if compress_mode >=3:   # ImageNet
            working_img_size *= 2
            self.decoder.append(nn.Conv2d(out_channels, 16, 3, 1, padding=1))
            self.decoder.append(nn.BatchNorm2d(16))
            if use_tanh:
                self.decoder.append(nn.Tanh())
            else:
                self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Upsample(scale_factor=2.0, mode="bilinear")) # ImageNet
            out_channels = 16
        if compress_mode >= 2:
            working_img_size *= 2
            self.decoder.append(nn.Conv2d(out_channels, 16, 3, 1, padding=1))
            self.decoder.append(nn.BatchNorm2d(16))
            if use_tanh:
                self.decoder.append(nn.Tanh())
            else:
                self.decoder.append(nn.ReLU())
            self.decoder.append(nn.Upsample(scale_factor=2.0, mode="nearest"))
        working_img_size *= 2
        self.decoder.append(nn.Conv2d(16, 16, 3, 1, padding=1))
        self.decoder.append(nn.BatchNorm2d(16))
        if use_tanh:
            self.decoder.append(nn.Tanh())
        else:
            self.decoder.append(nn.ReLU())
        self.decoder.append(nn.Upsample(scale_factor=2.0, mode="nearest"))
        if resize:
            self.decoder.append(nn.UpsamplingBilinear2d(img_size))
        self.decoder.append(nn.Conv2d(16, in_channels, 3, stride=1, padding=1))
        self.decoder.output_shape = (working_img_size, working_img_size)
        assert working_img_size == img_size if resize is None else resize
        self.encoder.forward = self.forward_encoder
        self.decoder.forward = self.forward_decoder

    def forward(self, x):
        for idx, m in enumerate(self.encoder):
            x = m(x)
        assert x.size(-1) == self.encoder.output_shape[-1]
        for idx, m in enumerate(self.decoder):
            x = m(x)
        return x

    def forward_encoder(self, x):
        for m in self.encoder:
            x = m(x)
        return x

    def forward_decoder(self, x):
        for m in self.decoder:
            x = m(x)
        return x

    def load_codec(self, weight_path):
        assert os.path.exists(weight_path), "{} must exist!".format(weight_path)
        loaded_dict = torch.load(weight_path, map_location=lambda storage, location: storage)
        self.encoder.load_state_dict(loaded_dict["encoder"])
        self.decoder.load_state_dict(loaded_dict["decoder"])
        log.info("Load codec model over.")
