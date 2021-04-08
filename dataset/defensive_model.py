import glob
import os

import pretrainedmodels
import torch
import torchvision.models as vision_models
from torch import nn
from torchvision import models as torch_models

import cifar_models as models
from adversarial_defense.com_defend.compression_network import ComDefend
from adversarial_defense.feature_distillation.jpeg import convert_images
from adversarial_defense.jpeg_compression.jpeg import JPEGFilter
from adversarial_defense.feature_scatter.attack_methods import Attack_FeaScatter
from adversarial_defense.model.denoise_resnet import DenoiseResNet50, DenoiseResNet101, DenoiseResNet152, \
    DenoiseResNet34
from adversarial_defense.model.guided_denoiser_network import Net
from adversarial_defense.model.pcl_resnet import PrototypeConformityLossResNet
from adversarial_defense.post_averaging.post_averaged_models import PostAveragedNetwork
from config import pretrained_cifar_model_conf, IN_CHANNELS, IMAGE_SIZE, CLASS_NUM, PY_ROOT
from tiny_imagenet_models.densenet import densenet161, densenet121, densenet169, densenet201
from tiny_imagenet_models.inception import inception_v3
from tiny_imagenet_models.resnext import resnext101_32x4d, resnext101_64x4d
from tiny_imagenet_models.wrn import tiny_imagenet_wrn
from adversarial_defense.model.resnet_return_feature import ResNet
from imagenet_models.resnet import resnet50 as resnet50_imagenet_adv_train
from adversarial_defense.model.wrn_return_feature import WideResNet
from adversarial_defense.model.tinyimagenet_resnet_return_feature import resnet101, resnet152, resnet34, resnet50

import glog as log

class DefensiveModel(nn.Module):
    """
    A DefensiveModel object wraps a cnn model.
    This model always accept standard image: in [0, 1] range, RGB order, un-normalized, NCHW format
    Two types of defensive models are available:
    1. com_defend, feature_distillation,post_averaging (based on preprocessing denoise)
    2. feature_scatter, pcl_loss, feature_denoise  (based on adversarial training)
    """
    def __init__(self, dataset, arch, no_grad, defense_model, norm="linf", eps="8_div_255"):
        super(DefensiveModel, self).__init__()
        # init cnn model
        self.in_channels = IN_CHANNELS[dataset]
        self.dataset = dataset
        self.arch = arch
        self.defense_model = defense_model
        self.num_classes = CLASS_NUM[dataset]

        if defense_model != "feature_denoise" and defense_model != "pcl_loss" and defense_model!="pcl_loss_adv_train" \
                and defense_model != "guided_denoiser" and defense_model != "TRADES" and defense_model!="adv_train" and defense_model!="adv_train_on_ImageNet":
            if dataset.startswith("CIFAR"):
                trained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}-pretrained/{arch}/checkpoint.pth.tar".format(root=PY_ROOT, dataset=dataset, arch=arch)
                assert os.path.exists(trained_model_path), "{} does not exist!".format(trained_model_path)
            elif dataset == "TinyImageNet":
                trained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}@{arch}@*.pth.tar".format(root=PY_ROOT, dataset=dataset, arch=arch)
                trained_model_path_list = list(glob.glob(trained_model_path))
                assert len(trained_model_path_list)>0, "{} does not exist!".format(trained_model_path)
                trained_model_path = trained_model_path_list[0]
            else:
                trained_model_path = "{root}/train_pytorch_model/real_image_model/{dataset}-pretrained/checkpoints/{arch}*.pth".format(
                    root=PY_ROOT, dataset=dataset, arch=arch)
                trained_model_path_ls = list(glob.glob(trained_model_path))
                assert trained_model_path_ls,  "{} does not exist!".format(trained_model_path)
                trained_model_path = trained_model_path_ls[0]

            self.model = self.make_model(dataset, arch, self.in_channels, CLASS_NUM[dataset],defense_model, trained_model_path=trained_model_path)
            # init cnn model meta-information
            self.mean = torch.FloatTensor(self.model.mean).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad =True
            self.std = torch.FloatTensor(self.model.std).view(1, self.in_channels, 1, 1).cuda()
            self.std.requires_grad = True
            self.input_space = self.model.input_space  # 'RGB' or 'GBR'
            self.input_range = self.model.input_range  # [0, 1] or [0, 255]
            self.input_size = self.model.input_size

        if defense_model == "com_defend":
            self.preprocessor = ComDefend(self.in_channels, 1.0)
            com_defend_model_path = "{}/train_pytorch_model/adversarial_train/com_defend/{}@*.pth.tar".format(PY_ROOT, dataset)
            com_defend_model_path = list(glob.glob(com_defend_model_path))[0]
            assert os.path.exists(com_defend_model_path), "{} does not exist!".format(com_defend_model_path)
            log.info('Using com_defend model from {}'.format(com_defend_model_path))
            self.preprocessor.load_state_dict(torch.load(com_defend_model_path, map_location=lambda storage, location: storage)["state_dict"])
            self.preprocessor.eval()
        elif defense_model == "feature_scatter":
            model_path = "{}/train_pytorch_model/adversarial_train/feature_scatter/{}@{}@*.pth.tar".format(PY_ROOT, dataset, arch)
            model_path = list(glob.glob(model_path))[0]
            config_feature_scatter = {
                'train': True,
                'epsilon': 8.0 / 255 * 2,
                'num_steps': 1,
                'step_size': 8.0 / 255 * 2,
                'random_start': True,
                'ls_factor': 0.5,
            }
            self.feature_scatter = Attack_FeaScatter(self.model, config_feature_scatter)
            state_dict = torch.load(model_path, map_location=lambda storage, location: storage)["net"]
            filtered_state_dict = {}
            for module, params in state_dict.items():
                filtered_state_dict[module.replace("module.","").replace("cnn.","")] = params
            self.feature_scatter.load_state_dict(filtered_state_dict)
            self.model = self.feature_scatter.basic_net
            log.info("Load feature scatter model from {}".format(model_path))
        elif defense_model == "TRADES":
            model_path = '{}/train_pytorch_model/adversarial_train/TRADES/{}@{}@norm_{}*.pth.tar'.format(
                PY_ROOT, dataset, arch, norm)
            log.info("accessing model path {}".format(model_path))
            model_path = list(glob.glob(model_path))[0]
            assert os.path.exists(model_path), "Model {} does not exist!".format(model_path)
            self.model = self.make_model(dataset, arch, self.in_channels, CLASS_NUM[dataset],defense_model,
                                         trained_model_path=model_path)
            # init cnn model meta-information
            self.mean = torch.FloatTensor(self.model.mean).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std = torch.FloatTensor(self.model.std).view(1, self.in_channels, 1, 1).cuda()
            self.std.requires_grad = True
            self.input_space = self.model.input_space  # 'RGB' or 'GBR'
            self.input_range = self.model.input_range  # [0, 1] or [0, 255]
            self.input_size = self.model.input_size
        elif defense_model == "adv_train":
            model_path = '{}/train_pytorch_model/adversarial_train/adv_train/{}@{}*.pth.tar'.format(
                PY_ROOT, dataset, arch)
            model_path = list(glob.glob(model_path))[0]
            assert os.path.exists(model_path), "Model {} does not exist!".format(model_path)
            self.model = self.make_model(dataset, arch, self.in_channels, CLASS_NUM[dataset],defense_model,
                                         trained_model_path=model_path)
            # init cnn model meta-information
            self.mean = torch.FloatTensor(self.model.mean).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std = torch.FloatTensor(self.model.std).view(1, self.in_channels, 1, 1).cuda()
            self.std.requires_grad = True
            self.input_space = self.model.input_space  # 'RGB' or 'GBR'
            self.input_range = self.model.input_range  # [0, 1] or [0, 255]
            self.input_size = self.model.input_size
        elif defense_model == "adv_train_on_ImageNet":
            model_path = '{}/train_pytorch_model/adversarial_train/adv_train/{}@{}@norm_{}@eps_{}.*'.format(
                PY_ROOT, dataset, arch, norm, eps)
            log.info(model_path)
            model_path = list(glob.glob(model_path))[0]
            assert os.path.exists(model_path), "Model {} does not exist!".format(model_path)
            self.model = self.make_model(dataset, arch, self.in_channels, CLASS_NUM[dataset],defense_model,
                                         trained_model_path=model_path)
            # init cnn model meta-information
            self.mean = torch.tensor([0.4850, 0.4560, 0.4060]).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std = torch.tensor([0.2290, 0.2240, 0.2250]).view(1, self.in_channels, 1, 1).cuda()
            self.std.requires_grad = True
            self.input_space = 'RGB'  # 'RGB' or 'GBR'
            self.input_range = [0, 1]  # [0, 1] or [0, 255]
            self.input_size = [3, 224, 224]
        elif defense_model == "LGD":
            model_path = "{}/train_pytorch_model/adversarial_train/guided_denoiser/guided_denoiser_{}_{}_LGD.pth.tar".format(PY_ROOT, dataset, arch)
            assert os.path.exists(model_path), "Model file {} does not exist!".format(model_path)
            checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
            if dataset.startswith("CIFAR"):
                if arch == "resnet-50":
                    classifier = ResNet(50, CLASS_NUM[dataset], block_name='BasicBlock')
                elif arch == "resnet-110":
                    classifier = ResNet(110, CLASS_NUM[dataset], block_name='BasicBlock')
                elif arch == "resnet101":
                    classifier = ResNet(56, CLASS_NUM[dataset], block_name='BasicBlock')
                elif arch == "WRN-28-10-drop":
                    classifier = WideResNet(28, CLASS_NUM[dataset], widen_factor=10, dropRate=0.3)
                elif arch == "WRN-40-10-drop":
                    classifier = WideResNet(40, CLASS_NUM[dataset], widen_factor=10, dropRate=0.3)
            elif dataset == "TinyImageNet":
                if arch == "resnet50":
                    classifier = resnet50(num_classes=CLASS_NUM[dataset],pretrained=False)
                elif arch == "resnet152":
                    classifier = resnet152(num_classes=CLASS_NUM[dataset],pretrained=False)
                elif arch == "resnet101":
                    classifier = resnet101(num_classes=CLASS_NUM[dataset],pretrained=False)
                elif arch == "WRN-28-10-drop":
                    classifier = WideResNet(28, CLASS_NUM[dataset], widen_factor=10, dropRate=0.3)
                elif arch == "WRN-40-10-drop":
                    classifier = WideResNet(40, CLASS_NUM[dataset], widen_factor=10, dropRate=0.3)
            net = Net(classifier, dataset, IMAGE_SIZE[dataset][0], IN_CHANNELS[dataset], 1, 0, False)
            net.load_state_dict(checkpoint['state_dict'])
            net.cuda()
            self.model = net
            self.mean = torch.FloatTensor([0, 0, 0]).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std = torch.FloatTensor([1, 1, 1]).view(1, self.in_channels, 1, 1).cuda()
            self.std.requires_grad = True
            self.input_space = 'RGB'
            self.input_range = [0, 1]
            self.input_size = [IN_CHANNELS[dataset], IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
        elif defense_model == "FGD":
            model_path = "{}/train_pytorch_model/adversarial_train/guided_denoiser/guided_denoiser_{}_{}_FGD.pth.tar".format(PY_ROOT, dataset, arch)
            assert os.path.exists(model_path), "Model file {} does not exist!".format(model_path)
            checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
            if dataset.startswith("CIFAR"):
                if arch == "resnet-50":
                    classifier = ResNet(50, CLASS_NUM[dataset], block_name='BasicBlock')
                elif arch == "resnet-110":
                    classifier = ResNet(110, CLASS_NUM[dataset], block_name='BasicBlock')
                elif arch == "resnet101":
                    classifier = ResNet(56, CLASS_NUM[dataset], block_name='BasicBlock')
                elif arch == "WRN-28-10-drop":
                    classifier = WideResNet(28, CLASS_NUM[dataset], widen_factor=10, dropRate=0.3)
                elif arch == "WRN-40-10-drop":
                    classifier = WideResNet(40, CLASS_NUM[dataset], widen_factor=10, dropRate=0.3)
            elif dataset == "TinyImageNet":
                if arch == "resnet50":
                    classifier = resnet50(num_classes=CLASS_NUM[dataset],pretrained=False)
                elif arch == "resnet152":
                    classifier = resnet152(num_classes=CLASS_NUM[dataset],pretrained=False)
                elif arch == "resnet101":
                    classifier = resnet101(num_classes=CLASS_NUM[dataset],pretrained=False)
                elif arch == "WRN-28-10-drop":
                    classifier = WideResNet(28, CLASS_NUM[dataset], widen_factor=10, dropRate=0.3)
                elif arch == "WRN-40-10-drop":
                    classifier = WideResNet(40, CLASS_NUM[dataset], widen_factor=10, dropRate=0.3)
            net = Net(classifier, dataset, IMAGE_SIZE[dataset][0], IN_CHANNELS[dataset], 1, 0, False)
            net.load_state_dict(checkpoint['state_dict'])
            net.cuda()
            self.model = net
            self.mean = torch.FloatTensor([0, 0, 0]).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std = torch.FloatTensor([1, 1, 1]).view(1, self.in_channels, 1, 1).cuda()
            self.std.requires_grad = True
            self.input_space = 'RGB'
            self.input_range = [0, 1]
            self.input_size = [IN_CHANNELS[dataset], IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
        elif defense_model == "guided_denoiser":
            model_path = "{}/train_pytorch_model/adversarial_train/guided_denoiser/guided_denoiser_{}_{}.pth.tar".format(PY_ROOT, dataset, arch)
            assert os.path.exists(model_path), "Model file {} does not exist!".format(model_path)
            checkpoint = torch.load(model_path, map_location=lambda storage, location: storage)
            if dataset.startswith("CIFAR"):
                if arch == "resnet-50":
                    classifier = ResNet(50, CLASS_NUM[dataset], block_name='BasicBlock')
                elif arch == "resnet-110":
                    classifier = ResNet(110, CLASS_NUM[dataset], block_name='BasicBlock')
                elif arch == "resnet101":
                    classifier = ResNet(56, CLASS_NUM[dataset], block_name='BasicBlock')
                elif arch == "WRN-28-10-drop":
                    classifier = WideResNet(28, CLASS_NUM[dataset], widen_factor=10, dropRate=0.3)
                elif arch == "WRN-40-10-drop":
                    classifier = WideResNet(40, CLASS_NUM[dataset], widen_factor=10, dropRate=0.3)
            elif dataset == "TinyImageNet":
                if arch == "resnet50":
                    classifier = resnet50(num_classes=CLASS_NUM[dataset],pretrained=False)
                elif arch == "resnet152":
                    classifier = resnet152(num_classes=CLASS_NUM[dataset],pretrained=False)
                elif arch == "resnet101":
                    classifier = resnet101(num_classes=CLASS_NUM[dataset],pretrained=False)
                elif arch == "WRN-28-10-drop":
                    classifier = WideResNet(28, CLASS_NUM[dataset], widen_factor=10, dropRate=0.3)
                elif arch == "WRN-40-10-drop":
                    classifier = WideResNet(40, CLASS_NUM[dataset], widen_factor=10, dropRate=0.3)
            net = Net(classifier, dataset, IMAGE_SIZE[dataset][0], IN_CHANNELS[dataset], 1, 0, False)
            net.load_state_dict(checkpoint['state_dict'])
            net.cuda()
            self.model = net
            self.mean = torch.FloatTensor([0, 0, 0]).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std = torch.FloatTensor([1, 1, 1]).view(1, self.in_channels, 1, 1).cuda()
            self.std.requires_grad = True
            self.input_space = 'RGB'
            self.input_range = [0, 1]
            self.input_size = [IN_CHANNELS[dataset], IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]

        elif defense_model == "feature_distillation":
            self.preprocessor = convert_images
        elif defense_model == "jpeg":
            self.jpeg_filter = JPEGFilter()
            self.preprocessor = self.jpeg_filter
        elif defense_model == "post_averaging":
            R = 30 if arch in ["resnet-152", "ResNet152","DenoiseResNet152"] else 6
            self.input_space = self.model.input_space  # 'RGB' or 'GBR'
            self.input_range = self.model.input_range  # [0, 1] or [0, 255]
            self.input_size = self.model.input_size
            self.model = PostAveragedNetwork(self.model, K=15, R=R, num_classes=CLASS_NUM[dataset]).cuda()
        elif defense_model == "pcl_loss":
            self.model = PrototypeConformityLossResNet(in_channels=IN_CHANNELS[dataset],
                                                       depth=pretrained_cifar_model_conf[dataset][arch]["depth"],
                                                       num_classes=CLASS_NUM[dataset])
            self.mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view(1, self.in_channels, 1, 1).cuda()
            self.std = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std.requires_grad = True
            self.input_space = 'RGB'
            self.input_range = [0, 1]
            self.input_size = [IN_CHANNELS[dataset], IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            pretrained_model_path = "{root}/train_pytorch_model/adversarial_train/pl_loss/pcl_train_{dataset}@{arch}.pth.tar".format(root=PY_ROOT, dataset=dataset, arch=arch)
            assert os.path.exists(pretrained_model_path), "{} does not exist!".format(pretrained_model_path)
            state_dict = torch.load(pretrained_model_path, map_location=lambda storage, location: storage)["state_dict"]
            filtered_dict = {}
            for key, value in state_dict.items():
                filtered_dict[key.replace("cnn.","")] = value
            self.model.load_state_dict(filtered_dict)
            log.info("Load pcl model from {} done".format(pretrained_model_path))
        elif defense_model == "pcl_loss_adv_train":
            self.model = PrototypeConformityLossResNet(in_channels=IN_CHANNELS[dataset], depth=pretrained_cifar_model_conf[dataset][arch]["depth"], num_classes=CLASS_NUM[dataset])
            self.mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view(1, self.in_channels, 1, 1).cuda()
            self.std = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std.requires_grad = True
            self.input_space = 'RGB'
            self.input_range = [0, 1]
            self.input_size = [IN_CHANNELS[dataset], IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            pretrained_model_path = "{root}/train_pytorch_model/adversarial_train/pl_loss/pcl_pgd_adv_train_{dataset}@{arch}.pth.tar".format(root=PY_ROOT, dataset=dataset, arch=arch)
            assert os.path.exists(pretrained_model_path), "{} does not exist!".format(pretrained_model_path)
            state_dict = torch.load(pretrained_model_path, map_location=lambda storage, location: storage)["state_dict"]
            filtered_dict = {}
            for key, value in state_dict.items():
                filtered_dict[key.replace("cnn.","")] = value
            self.model.load_state_dict(filtered_dict)
            log.info("Load pcl model from {} done".format(pretrained_model_path))
        elif defense_model == "feature_denoise":
            filter_type = "NonLocal_Filter"
            ksize = 3
            if arch == "DenoiseResNet34":
                self.model = DenoiseResNet34(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset],
                                          whether_denoising=True, filter_type=filter_type, ksize=ksize)
            elif arch == "DenoiseResNet50":
                self.model = DenoiseResNet50(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset],
                                          whether_denoising=True, filter_type=filter_type, ksize=ksize)
            elif arch == "DenoiseResNet101":
                self.model = DenoiseResNet101(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset],
                                       whether_denoising=True, filter_type=filter_type, ksize=ksize)
            elif arch == "DenoiseResNet152":
                self.model = DenoiseResNet152(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset],
                                       whether_denoising=True, filter_type=filter_type, ksize=ksize)

            self.mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view(1, self.in_channels, 1, 1).cuda()
            self.std = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std.requires_grad = True
            self.input_space = 'RGB'
            self.input_range = [0, 1]
            self.input_size = [IN_CHANNELS[dataset], IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            pretrained_model_path = "{root}/train_pytorch_model/adversarial_train/feature_denoise/{dataset}@{arch}_{filter_type}_{ksize}.pth.tar".format(
                root=PY_ROOT, dataset=dataset, arch=arch, filter_type=filter_type, ksize=ksize)
            assert os.path.exists(pretrained_model_path), "{} does not exist!".format(pretrained_model_path)
            state_dict = torch.load(pretrained_model_path, map_location=lambda storage, location: storage)["state_dict"]
            log.info("Load feature denoise model from {} done".format(pretrained_model_path))
            filtered_dict = {}
            for key, value in state_dict.items():
                filtered_dict[key.replace("cnn.", "")] = value
            self.model.load_state_dict(filtered_dict)
        elif defense_model == "feature_denoise_adv_train":
            filter_type = "NonLocal_Filter"
            ksize = 3
            if arch == "DenoiseResNet34":
                self.model = DenoiseResNet34(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset],
                                             whether_denoising=True, filter_type=filter_type, ksize=ksize)
            elif arch == "DenoiseResNet50":
                self.model = DenoiseResNet50(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset],
                                             whether_denoising=True, filter_type=filter_type, ksize=ksize)
            elif arch == "DenoiseResNet101":
                self.model = DenoiseResNet101(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset],
                                              whether_denoising=True, filter_type=filter_type, ksize=ksize)
            elif arch == "DenoiseResNet152":
                self.model = DenoiseResNet152(in_channels=IN_CHANNELS[dataset], num_classes=CLASS_NUM[dataset],
                                              whether_denoising=True, filter_type=filter_type, ksize=ksize)

            self.mean = torch.FloatTensor([0.4914, 0.4822, 0.4465]).view(1, self.in_channels, 1, 1).cuda()
            self.std = torch.FloatTensor([0.2023, 0.1994, 0.2010]).view(1, self.in_channels, 1, 1).cuda()
            self.mean.requires_grad = True
            self.std.requires_grad = True
            self.input_space = 'RGB'
            self.input_range = [0, 1]
            self.input_size = [IN_CHANNELS[dataset], IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            pretrained_model_path = "{root}/train_pytorch_model/adversarial_train/feature_denoise/pgd_adv_train_{dataset}@{arch}_NonLocal_Filter_3.pth.tar".format(
                root=PY_ROOT, dataset=dataset, arch=arch)
            assert os.path.exists(pretrained_model_path), "{} does not exist!".format(pretrained_model_path)
            state_dict = torch.load(pretrained_model_path, map_location=lambda storage, location: storage)["state_dict"]
            filtered_dict = {}
            for key, value in state_dict.items():
                filtered_dict[key.replace("cnn.", "")] = value
            self.model.load_state_dict(filtered_dict)
            log.info("Load feature denoise model from {} done".format(pretrained_model_path))

        self.no_grad = no_grad

    def forward(self, x):
        # assign dropout probability
        # channel order
        if self.input_space == 'BGR':
            x = x[:, [2, 1, 0], :, :]  # pytorch does not support negative stride index (::-1) yet
        # input range
        if max(self.input_range) == 255:
            x = x * 255
        if self.no_grad:
            with torch.no_grad():
                if hasattr(self, "preprocessor"):
                    x = self.preprocessor(x) # feature distillation
                # normalization
                x = (x - self.mean.type(x.dtype).to(x.device)) / self.std.type(x.dtype).to(x.device)
                if self.defense_model == "guided_denoiser" or self.defense_model == "LGD" or self.defense_model == "FGD":
                    x = self.model(None, x, requires_control=False, train=False)
                elif self.defense_model == "pcl_loss" or self.defense_model == "pcl_loss_adv_train":
                    x = self.model(x)[-1]
                else:
                    x = self.model(x)
        else:
            if hasattr(self, "preprocessor"):
                x = self.preprocessor(x)  # feature distillation
            # normalization
            x = (x - self.mean.type(x.dtype).to(x.device)) / self.std.type(x.dtype).to(x.device)
            if self.defense_model == "guided_denoiser" or self.defense_model == "LGD" or self.defense_model == "FGD":
                x = self.model(None, x, requires_control=False, train=False)
            elif self.defense_model == "pcl_loss" or self.defense_model == "pcl_loss_adv_train":
                x = self.model(x)[-1]
            else:
                x = self.model(x)
        x = x.view(x.size(0), -1)
        return x

    def load_weight_from_pth_checkpoint(self, model, fname):
        raw_state_dict = torch.load(fname, map_location='cpu')['state_dict']
        state_dict = dict()
        for key, val in raw_state_dict.items():
            new_key = key.replace('module.', '').replace('cnn.', "")  # FIXME why cnn?
            state_dict[new_key] = val
        model.load_state_dict(state_dict)
    def load_weight_from_ImageNet_adv_train_pth_checkpoint(self, model, fname):
        raw_state_dict = torch.load(fname, map_location='cpu')['state_dict']
        state_dict = dict()
        for key, val in raw_state_dict.items():
            new_key = key.replace('module.model.', '')  # FIXME why cnn?
            new_key = new_key.replace('module.attacker.model.','')
            state_dict[new_key] = val
        model.load_state_dict(state_dict,strict=False)

    def construct_cifar_model(self, arch, dataset, num_classes):
        conf = pretrained_cifar_model_conf[dataset][arch]
        arch = arch.split("-")[0].lower()
        if arch.startswith('resnext'):
            model = models.__dict__[arch](
                cardinality=conf["cardinality"],
                num_classes=num_classes,
                depth=conf["depth"],
                widen_factor=conf["widen_factor"],
                dropRate=conf["drop"],
            )
        elif arch.startswith('densenet'):
            model = models.__dict__[arch](
                num_classes=num_classes,
                depth=conf["depth"],
                growthRate=conf["growthRate"],
                compressionRate=conf["compressionRate"],
                dropRate=conf["drop"],
            )
        elif arch.startswith('wrn'):
            model = models.__dict__[arch](
                num_classes=num_classes,
                depth=conf["depth"],
                widen_factor=conf["widen_factor"],
                dropRate=conf["drop"],
            )
        elif arch.endswith('resnet'):
            model = models.__dict__[arch](
                num_classes=num_classes,
                depth=conf["depth"],
                block_name=conf["block_name"],
            )
        else:
            model = models.__dict__[arch](num_classes=num_classes)
        return model

    def make_model(self, dataset, arch, in_channel, num_classes, defense_model, trained_model_path=None):
        """
        Make model, and load pre-trained weights.
        :param dataset: cifar10 or imagenet
        :param arch: arch name, e.g., alexnet_bn
        :return: model (in cpu and training mode)
        """
        if dataset in ['CIFAR-10',"CIFAR-100", "MNIST","FashionMNIST"]:
            assert trained_model_path is not None and os.path.exists(trained_model_path), "Pretrained weight model file {} does not exist!".format(trained_model_path)
            if arch == 'gdas':
                model = models.gdas(in_channel, num_classes)
                model.mean = [125.3 / 255, 123.0 / 255, 113.9 / 255]
                model.std = [63.0 / 255, 62.1 / 255, 66.7 / 255]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size = [in_channel, IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            elif arch == 'pyramidnet272':
                model = models.pyramidnet272(in_channel, num_classes)
                model.mean = [0.49139968, 0.48215841, 0.44653091]
                model.std = [0.24703223, 0.24348513, 0.26158784]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size = [in_channel, IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            else:
                model = self.construct_cifar_model(arch, dataset, num_classes)
                model.mean = [0.4914, 0.4822, 0.4465]
                model.std = [0.2023, 0.1994, 0.2010]
                model.input_space = 'RGB'
                model.input_range = [0, 1]
                model.input_size = [in_channel, IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            self.load_weight_from_pth_checkpoint(model, trained_model_path)
            log.info("load weights of {} from {} successfully!".format(arch, trained_model_path))
        elif dataset == "TinyImageNet":
            model = MetaLearnerModelBuilder.construct_tiny_imagenet_model(arch, dataset)
            model.input_space = 'RGB'
            model.input_range = [0, 1]
            model.mean = [0,0,0]
            model.std = [1,1,1]
            model.input_size = [in_channel,IMAGE_SIZE[dataset][0], IMAGE_SIZE[dataset][1]]
            self.load_weight_from_pth_checkpoint(model, trained_model_path)
            log.info("load weights of {} from {} successfully!".format(arch, trained_model_path))
            # model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, location: storage)["state_dict"])
        elif dataset == 'ImageNet':
            if arch == "resnet50" and defense_model == "adv_train_on_ImageNet":
                model = resnet50_imagenet_adv_train(pretrained=False,progress=True)
                self.load_weight_from_ImageNet_adv_train_pth_checkpoint(model,trained_model_path)
                log.info("loading adv train model {}".format(trained_model_path))
            else:
                os.environ["TORCH_HOME"] = "{}/train_pytorch_model/real_image_model/ImageNet-pretrained".format(PY_ROOT)
                model = pretrainedmodels.__dict__[arch](num_classes=1000, pretrained="imagenet")
        return model


class MetaLearnerModelBuilder(object):

    @staticmethod
    def construct_imagenet_model(arch, dataset):
        os.environ["TORCH_HOME"] = "{}/train_pytorch_model/real_image_model/ImageNet-pretrained".format(PY_ROOT)
        model = vision_models.__dict__[arch](pretrained=False)
        return model

    @staticmethod
    def construct_tiny_imagenet_model(arch, dataset):
        if not arch.startswith("densenet") and not arch.startswith("resnext") and arch in torch_models.__dict__:
            network = torch_models.__dict__[arch](pretrained=False)
        num_classes = CLASS_NUM[dataset]
        if arch.startswith("resnet"):
            num_ftrs = network.fc.in_features
            network.fc = nn.Linear(num_ftrs, num_classes)
        elif arch.startswith("densenet"):
            if arch == "densenet161":
                network = densenet161(pretrained=False)
            elif arch == "densenet121":
                network = densenet121(pretrained=False)
            elif arch == "densenet169":
                network = densenet169(pretrained=False)
            elif arch == "densenet201":
                network = densenet201(pretrained=False)
        elif arch == "resnext32_4":
            network = resnext101_32x4d(pretrained=None)
        elif arch == "resnext64_4":
            network = resnext101_64x4d(pretrained=None)
        elif arch.startswith("inception"):
            network = inception_v3(pretrained=False)
        elif arch == "WRN-28-10-drop":
            network = tiny_imagenet_wrn(in_channels=IN_CHANNELS[dataset],depth=28,num_classes=CLASS_NUM[dataset],widen_factor=10, dropRate=0.3)
        elif arch == "WRN-40-10-drop":
            network = tiny_imagenet_wrn(in_channels=IN_CHANNELS[dataset], depth=40, num_classes=CLASS_NUM[dataset],
                                        widen_factor=10, dropRate=0.3)
        elif arch.startswith("vgg"):
            network.avgpool = Identity()
            network.classifier[0] = nn.Linear(512 * 2 * 2, 4096)  # 64 /2**5 = 2
            network.classifier[-1] = nn.Linear(4096, num_classes)
        return network



