import torch
from meta_attack.meta_training.learner import Learner
from torch import nn
config = [
        ('conv2d', [32, 3, 3, 3, 1, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('conv2d', [64, 32, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('conv2d', [128, 64, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [128]),
        ('conv2d', [256, 128, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [256]),
        ('convt2d', [256, 128, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [128]),
        ('convt2d', [128, 64, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [64]),
        ('convt2d', [64, 32, 4, 4, 2, 1]),
        ('relu', [True]),
        ('bn', [32]),
        ('convt2d', [32, 3, 3, 3, 1, 1]),
   ]

class Meta(nn.Module):
    def __init__(self):
        super(Meta, self).__init__() 
        self.net = Learner(config)
    def forward(self,x):
        return x

def load_meta_model(meta_model_path):
    # meta_model = Learner(config, 3, 32)
    meta_model = Meta()
    pretrained_dict = torch.load(meta_model_path, map_location=lambda storage, location: storage)["state_dict"]
    meta_model.load_state_dict(pretrained_dict, strict=True)
    meta_model.net.eval()
    meta_model.cuda()
    return meta_model.net.cuda()


