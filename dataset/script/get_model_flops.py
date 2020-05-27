from dataset.standard_model import StandardModel
from torchstat import stat
from thop import profile
from thop import clever_format

import torch
import numpy  as np

models_CIFAR10 = {"pyramidnet272": StandardModel("CIFAR-10", "pyramidnet272", no_grad=True),
          "gdas": StandardModel("CIFAR-10", "gdas", no_grad=True),
            "WRN-28": StandardModel("CIFAR-10", "WRN-28-10-drop", no_grad=True),
        "WRN-40": StandardModel("CIFAR-10", "WRN-40-10-drop", no_grad=True),
         }


models_CIFAR100 = {"pyramidnet272": StandardModel("CIFAR-100", "pyramidnet272", no_grad=True),
          "gdas": StandardModel("CIFAR-100", "gdas", no_grad=True),
            "WRN-28": StandardModel("CIFAR-100", "WRN-28-10-drop", no_grad=True),
        "WRN-40": StandardModel("CIFAR-100", "WRN-40-10-drop", no_grad=True),
         }


models_tiny_imagenet = {
    "densenet-121": StandardModel("TinyImageNet", "densenet121", no_grad=True),
    "resnext32_4": StandardModel("TinyImageNet", "resnext32_4", no_grad=True),
    "resnext64_4": StandardModel("TinyImageNet", "resnext64_4", no_grad=True)
}

def params_count(model):
    """
    Compute the number of parameters.
    Args:
        model (model): model to count the number of parameters.
    """
    return np.sum([p.numel() for p in model.parameters()]).item()


# for model_name, model in models.items():
#     stat(model,(3, 32, 32), query_granularity=1)
#     params = params_count(model)
#     print(model_name + " done! params:{}".format(params))

    # print("Model:{}, FLOPS:{}, Params:{}".format(model_name, flops, params))

for model_name, model in models_CIFAR100.items():
    input = torch.randn(1, 3, 32, 32)
    macs, params = profile(model, inputs=(input, ))
    # macs, params = clever_format([macs, params], "%.3f")
    print("%s params: %.2f flops: %.2f" % (model_name, params / (1000 ** 2), macs / (1000 ** 3)))

    # print("Model:{}, FLOPS:{}, Params:{}".format(model_name, flops, params))
