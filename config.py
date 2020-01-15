IMAGE_SIZE = {"CIFAR-10":(32,32), "ImageNet":(224,224), "MNIST":(28, 28), "FashionMNIST":(28,28), "SVHN":(32,32), "TinyImageNet": (64,64)}
IN_CHANNELS = {"MNIST":1, "FashionMNIST":1, "CIFAR-10":3, "ImageNet":3, "CIFAR-100":3, "SVHN":3, "TinyImageNet":3}
CLASS_NUM = {"MNIST":10,"FashionMNIST":10, "CIFAR-10":10, "CIFAR-100":100, "ImageNet":1000, "SVHN":10, "TinyImageNet":200}
PY_ROOT = "/home1/machen/meta_perturbations_black_box_attack"

IMAGE_DATA_ROOT = {"CIFAR-10":"/home1/machen/dataset/CIFAR-10", 'cifar10':"/home1/machen/dataset/CIFAR-10",
                   "MNIST":"/home1/machen/dataset/MNIST",
                   "FashionMNIST":"/home1/machen/dataset/FashionMNIST",
                   "SVHN":"/home1/machen/dataset/SVHN",
                   "ImageNet": "/home1/machen/dataset/ILSVRC2012/",
                   "TinyImageNet": "/home1/machen/dataset/tinyImageNet/"}

MODELS_TRAIN = ['conv3', 'densenet121', 'densenet161', 'densenet169', 'densenet201', 'dpn26', 'dpn92', 'efficientnet', 'googlenet',
            'mobilenet', 'mobilenet_v2', 'pnasnetA', 'pnasnetB', 'preactresnet101', 'preactresnet152', 'preactresnet18',
            'preactresnet34', 'preactresnet50', 'resnext29_2', 'resnext29_32', 'resnext29_4', 'resnext29_8', 'resnext32_4',
            'resnext64_4', 'senet18', "resnext101_32x8d","resnext50_32x4d",
            'shufflenet_G2', 'shufflenet_G3', 'shufflenet_v2_x0_5','shufflenet_v2_x1_0', 'squeezenet1_0', 'squeezenet1_1',
            'vgg11', 'vgg13', 'vgg16', 'vgg19','vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn']
MODELS_TEST = ["inception_v3", "wideresnet28drop","wideresnet28","gdas", "pyramidnet272", "carlinet"
               "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",  "wideresnet34", "wideresnet40",
                "wideresnet34drop", "wideresnet40drop"]


ALL_MODELS= MODELS_TRAIN + MODELS_TEST

MAX_QUERIES_GENERATE_TRAIN_DATA = {"MNIST": 200, "CIFAR-10":2000, "FashionMNIST":200}