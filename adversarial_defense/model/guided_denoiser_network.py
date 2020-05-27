from torch import nn
import torch
import numpy as np

class DenoiseLoss(nn.Module):
    def __init__(self, n, hard_mining = 0, norm = False):
        super(DenoiseLoss, self).__init__()
        self.n = n
        assert(hard_mining >= 0 and hard_mining <= 1)
        self.hard_mining = hard_mining
        self.norm = norm

    def forward(self, x, y):
        loss = torch.pow(torch.abs(x - y), self.n) / self.n
        if self.hard_mining > 0:
            loss = loss.view(-1)
            k = int(loss.size(0) * self.hard_mining)
            loss, idcs = torch.topk(loss, k)
            y = y.view(-1)[idcs]

        loss = loss.mean()
        if self.norm:
            norm = torch.pow(torch.abs(y), self.n)
            norm = norm.data.mean()
            loss = loss / norm
        return loss

class Loss(nn.Module):
    def __init__(self, n, hard_mining = 0, norm = False):
        super(Loss, self).__init__()
        self.loss = DenoiseLoss(n, hard_mining, norm)

    def forward(self, x, y):
        z = []
        for i in range(len(x)):
            z.append(self.loss(x[i], y[i]))
        return z

class ConvBNReLU(nn.Module):
    def __init__(self, n_in, n_out, stride = 1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(n_in, n_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn = nn.BatchNorm2d(n_out)
        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, n_in, n_out, stride = 1, expansion = 4):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, kernel_size = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.conv2 = nn.Conv2d(n_out, n_out, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(n_out)
        self.conv3 = nn.Conv2d(n_out, n_out * expansion, kernel_size = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(n_out * expansion)

        self.downsample = None
        if stride != 1 or n_in != n_out * expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(n_in, n_out * expansion, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(n_out * expansion))

        self.relu = nn.ReLU(inplace = True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Denoise(nn.Module):
    def __init__(self, h_in, w_in, block, fwd_in, fwd_out, num_fwd, back_out, num_back):
        super(Denoise, self).__init__()

        h, w = [], []
        for i in range(len(num_fwd)):
            h.append(h_in)
            w.append(w_in)
            h_in = int(np.ceil(float(h_in) / 2))
            w_in = int(np.ceil(float(w_in) / 2))

        if block is Bottleneck:
            expansion = 4
        else:
            expansion = 1

        fwd = []
        n_in = fwd_in
        for i in range(len(num_fwd)):
            group = []
            for j in range(num_fwd[i]):
                if j == 0:
                    if i == 0:
                        stride = 1
                    else:
                        stride = 2
                    group.append(block(n_in, fwd_out[i], stride=stride))
                else:
                    group.append(block(fwd_out[i] * expansion, fwd_out[i]))
            n_in = fwd_out[i] * expansion
            fwd.append(nn.Sequential(*group))
        self.fwd = nn.ModuleList(fwd)

        upsample = []
        back = []
        n_in = (fwd_out[-2] + fwd_out[-1]) * expansion
        for i in range(len(num_back) - 1, -1, -1):
            upsample.insert(0, nn.Upsample(size=(h[i], w[i]), mode='bilinear',align_corners=True))
            group = []
            for j in range(num_back[i]):
                if j == 0:
                    group.append(block(n_in, back_out[i]))
                else:
                    group.append(block(back_out[i] * expansion, back_out[i]))
            if i != 0:
                n_in = (back_out[i] + fwd_out[i - 1]) * expansion
            back.insert(0, nn.Sequential(*group))
        self.upsample = nn.ModuleList(upsample)
        self.back = nn.ModuleList(back)

        self.final = nn.Conv2d(back_out[0] * expansion, fwd_in, kernel_size=1, bias=False)

    def forward(self, x):
        out = x
        outputs = []
        for i in range(len(self.fwd)):
            out = self.fwd[i](out)
            if i != len(self.fwd) - 1:
                outputs.append(out)

        for i in range(len(self.back) - 1, -1, -1):
            out = self.upsample[i](out)
            out = torch.cat((out, outputs[i]), 1)
            out = self.back[i](out)
        out = self.final(out)
        out += x
        return out


class Net(nn.Module):
    def __init__(self, classifier, dataset, img_size, in_channels, n, hard_mining=0, loss_norm=False):
        super(Net, self).__init__()
        fwd_out = [64, 128, 256, 256, 256]
        num_fwd = [2, 3, 3, 3, 3]
        back_out = [64, 128, 256, 256]
        num_back = [2, 3, 3, 3]
        block = ConvBNReLU
        self.denoise = Denoise(img_size, img_size, block, in_channels, fwd_out, num_fwd, back_out, num_back)
        self.net = classifier  # 这种network，必须能返回最后一层卷积层输出
        self.loss = Loss(n, hard_mining, loss_norm)
        self.in_channels = in_channels
        if dataset.startswith("CIFAR"):
            self.mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view(1, self.in_channels, 1, 1).cuda()
            self.std.requires_grad = True
        else:
            self.mean = torch.FloatTensor([0,0,0]).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std = torch.FloatTensor([1,1,1]).view(1, self.in_channels, 1, 1).cuda()
            self.std.requires_grad = True

    def forward(self, orig_x, adv_x, requires_control=True, train=True):
        if train:
            orig_x = (orig_x - self.mean.type(orig_x.dtype).to(orig_x.device)) / self.std.type(orig_x.dtype).to(orig_x.device)
            adv_x = (adv_x - self.mean.type(adv_x.dtype).to(adv_x.device)) / self.std.type(adv_x.dtype).to(adv_x.device)
            orig_outputs = self.net(orig_x)
            orig_outputs.insert(0, orig_x)

            if requires_control:
                control_outputs = self.net(adv_x)
                control_outputs.insert(0, adv_x)
                control_loss = self.loss(control_outputs, orig_outputs)

            # if train:
            #     adv_x.requires_grad = True
            #     for i in range(len(orig_outputs)):
            #         orig_outputs[i].requires_grad = True
            adv_x = self.denoise(adv_x)
            adv_outputs = self.net(adv_x)
            adv_outputs.insert(0, adv_x)
            loss = self.loss(adv_outputs, orig_outputs)

            if not requires_control:
                return orig_outputs[-1], adv_outputs[-1], loss
            else:
                return orig_outputs[-1], adv_outputs[-1], loss, control_outputs[-1], control_loss
        else:
            assert orig_x is None
            assert requires_control == False
            adv_x = (adv_x - self.mean.type(adv_x.dtype).to(adv_x.device)) / self.std.type(adv_x.dtype).to(adv_x.device)
            adv_x = self.denoise(adv_x)
            adv_outputs = self.net(adv_x)
            return adv_outputs[-1]