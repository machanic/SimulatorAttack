import torch.nn as nn
import torch
import numpy as np
import torchvision.models as models
from torch.nn import functional as F

def perturb_image(image, noise):
    c, h, w = image.size()
    adv_image = image + F.interpolate(noise.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=True).squeeze(0)
    adv_image = torch.clamp(adv_image, 0, 1)
    return adv_image


def flip_noise(noise, block, block_size):
    noise_new = noise.clone()
    x, y, c = block[0:3]
    x = int(x*block_size)
    y = int(y*block_size)
    c = int(c)
    noise_new[c, x:x+block_size, y:y+block_size] *= -1
    return noise_new


def change_noise(noise, block, block_size, sigma, epsilon):
    noise_new = noise.clone()
    x, y, c = block[0:3]
    x = int(x*block_size)
    y = int(y*block_size)
    c = int(c)
    noise_new[c, x:x+block_size, y:y+block_size] += sigma
    noise_new = torch.clamp(noise_new, -epsilon, epsilon)
    return noise_new


class MarginLoss(nn.Module):

    def __init__(self, margin=1.0, target=False):
        super(MarginLoss, self).__init__()
        self.margin = margin
        self.target = target

    def forward(self, logits, label):

        if not self.target:
            one_hot= torch.zeros_like(logits, dtype=torch.uint8)
            label = label.reshape(-1,1)
            one_hot.scatter_(1, label, 1)
            diff = logits[one_hot] - torch.max(logits[~one_hot].view(len(logits),-1), dim=1)[0]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        else:
            diff = torch.max(torch.cat((logits[:, :label],logits[:,(label+1):]), dim=1), dim=1)[0] - logits[:, label]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        return margin


class Function(nn.Module):

    def __init__(self, model, batch_size=256, margin=0, nlabels=10, target=False):
        super(Function, self).__init__()
        self.model = model
        self.margin = margin
        self.target = target
        self.batch_size = batch_size
        self.current_counts = 0
        self.counts = []
        self.nlabels = nlabels

    def _loss(self, logits, label):
        if not self.target:
            if label == 0:
                logits_cat = logits[:,(label+1):]
            elif label == logits.size()[1] - 1:
                logits_cat = logits[:, :label]
            else:
                logits_cat = torch.cat((logits[:, :label],logits[:,(label+1):]), dim=1)
            diff = logits[:,label] - torch.max(logits_cat, dim=1)[0]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        else:
            diff = torch.max(torch.cat((logits[:, :label],logits[:,(label+1):]), dim=1), dim=1)[0] - logits[:, label]
            margin = torch.nn.functional.relu(diff + self.margin, True) - self.margin
        return margin

    def forward(self, images, label):
        if len(images.size())==3:
            images = images.unsqueeze(0)
        n = len(images)
        device = images.device
        k = 0
        loss = torch.zeros(n, dtype=torch.float32, device=device)
        logits = torch.zeros((n, self.nlabels), dtype=torch.float32, device=device)

        while k < n:
            start = k
            end = min(k + self.batch_size, n)
            logits[start:end] = self.model(images[start:end])
            loss[start:end] = self._loss(logits[start:end], label)
            k = end
        self.current_counts += n

        return logits, loss

    def new_counter(self):
        if self.current_counts != 0:
            self.counts.append(self.current_counts)
        self.current_counts = 0

    def update_counter(self, index):
        self.counts[index] = self.current_counts
        self.current_counts = 0

    def get_average(self, iter=10000):
        counts = np.array(self.counts)
        return np.mean(counts[counts<iter])


def get_model(model_name):

    if model_name == 'Resnet50':
        pretrained_model = models.resnet50(pretrained=True)
    elif model_name == 'VGG16':
        pretrained_model = models.vgg16_bn(pretrained=True)
    elif model_name == 'Densenet121':
        pretrained_model = models.densenet121(pretrained=True)


    return pretrained_model

