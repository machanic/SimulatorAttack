import os

import cv2
import shutil
import torch
import torch.nn.functional as F
import torchvision
import numpy as np
from torchvision import transforms

from config import MODELS_TEST_STANDARD


def find_imagenet_layer(arch, target_layer_name):
    if target_layer_name.startswith("layer"):  # senet154
        target_layer = arch.layer4
    elif target_layer_name.startswith("cell"): # pnasnet5large
        target_layer = arch.cell_11
    elif target_layer_name == "feature":
        target_layer = arch._modules[target_layer_name]  # the layer is "features", "layer4" of senet154

    return target_layer

def find_imagenet_inceptionv3_layer(arch, target_layer_name):
    # Mixed_7c
    return arch.cnn._modules[target_layer_name]

def find_imagenet_inceptionv4_layer(arch, target_layer_name):
    # features
    return arch.cnn._modules[target_layer_name]

def find_imagenet_senet154_layer(arch, target_layer_name):
    return arch.cnn.layer4

def find_imagenet_pnasnet5large(arch, target_layer_name):
    return arch.cnn.cell_11

def find_pyramidnet_layer(arch, target_layer_name):
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer




def find_squeezenet_layer(arch, target_layer_name):
    """Find squeezenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features_12'
            target_layer_name = 'features_12_expand3x3'
            target_layer_name = 'features_12_expand3x3_activation'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) == 3:
        target_layer = target_layer._modules[hierarchy[2]]

    elif len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[2] + '_' + hierarchy[3]]

    return target_layer

def find_resnet_layer(arch, target_layer_name):
    """Find resnet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'conv1'
            target_layer_name = 'layer1'
            target_layer_name = 'layer1_basicblock0'
            target_layer_name = 'layer1_basicblock0_relu'
            target_layer_name = 'layer1_bottleneck0'
            target_layer_name = 'layer1_bottleneck0_conv1'
            target_layer_name = 'layer1_bottleneck0_downsample'
            target_layer_name = 'layer1_bottleneck0_downsample_0'
            target_layer_name = 'avgpool'
            target_layer_name = 'fc'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """
    if 'layer' in target_layer_name:
        hierarchy = target_layer_name.split('_')
        layer_num = int(hierarchy[0].lstrip('layer'))
        if layer_num == 1:
            target_layer = arch.layer1
        elif layer_num == 2:
            target_layer = arch.layer2
        elif layer_num == 3:
            target_layer = arch.layer3
        elif layer_num == 4:
            target_layer = arch.layer4
        else:
            raise ValueError('unknown layer : {}'.format(target_layer_name))

        if len(hierarchy) >= 2:
            bottleneck_num = int(hierarchy[1].lower().lstrip('bottleneck').lstrip('basicblock'))
            target_layer = target_layer[bottleneck_num]

        if len(hierarchy) >= 3:
            target_layer = target_layer._modules[hierarchy[2]]

        if len(hierarchy) == 4:
            target_layer = target_layer._modules[hierarchy[3]]

    else:
        target_layer = arch._modules[target_layer_name]

    return target_layer


def find_densenet_layer(arch, target_layer_name):
    """Find densenet layer to calculate GradCAM and GradCAM++

    Args:
        arch: default torchvision densenet models
        target_layer_name (str): the name of layer with its hierarchical information. please refer to usages below.
            target_layer_name = 'features'
            target_layer_name = 'features_transition1'
            target_layer_name = 'features_transition1_norm'
            target_layer_name = 'features_denseblock2_denselayer12'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'features_denseblock2_denselayer12_norm1'
            target_layer_name = 'classifier'

    Return:
        target_layer: found layer. this layer will be hooked to get forward/backward pass information.
    """

    hierarchy = target_layer_name.split('_')
    target_layer = arch._modules[hierarchy[0]]

    if len(hierarchy) >= 2:
        target_layer = target_layer._modules[hierarchy[1]]

    if len(hierarchy) >= 3:
        target_layer = target_layer._modules[hierarchy[2]]

    if len(hierarchy) == 4:
        target_layer = target_layer._modules[hierarchy[3]]

    return target_layer



class GradCAM(object):
    """Calculate GradCAM salinecy map.
    A simple example:
        # initialize a model, model_dict and gradcam
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcam = GradCAM(model_dict)
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalize(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcam(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

    def __init__(self, model_dict, verbose=False):
        model_type = model_dict['model_type']
        layer_name = model_dict['layer_name']
        self.model_arch = model_dict['arch']

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None

        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None

        if 'resnet' in model_type.lower():
            target_layer = find_resnet_layer(self.model_arch, layer_name)
        elif 'squeezenet' in model_type.lower():
            target_layer = find_squeezenet_layer(self.model_arch, layer_name)
        elif "pyramidnet" in model_type.lower():
            target_layer = find_pyramidnet_layer(self.model_arch, layer_name)
        elif "senet" in model_type.lower():
            target_layer = find_imagenet_senet154_layer(self.model_arch, layer_name)
        elif "pnasnet5large" in model_type.lower():
            target_layer = find_imagenet_pnasnet5large(self.model_arch, layer_name)
        elif "inceptionv3" in model_type.lower():
            target_layer = find_imagenet_inceptionv3_layer(self.model_arch, layer_name)
        elif "inceptionv4" in model_type.lower():
            target_layer = find_imagenet_inceptionv4_layer(self.model_arch, layer_name)

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

        if verbose:
            try:
                input_size = model_dict['input_size']
            except KeyError:
                print("please specify size of input image in model_dict. e.g. {'input_size':(224, 224)}")
                pass
            else:
                device = 'cuda' if next(self.model_arch.parameters()).is_cuda else 'cpu'
                self.model_arch(torch.zeros(1, 3, *(input_size), device=device))
                print('saliency_map size :', self.activations['value'].shape[2:])

    def visualize_cam(self, mask, img):
        """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
        Args:
            mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
            img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

        Return:
            heatmap (torch.tensor): heatmap img shape of (3, H, W)
            result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * mask.detach().cpu().squeeze()), cv2.COLORMAP_JET)
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
        # b, g, r = heatmap.split(1)
        # heatmap = torch.cat([r, g, b])

        result = heatmap + img.detach().cpu()
        result = result.div(result.max()).squeeze()
        result = (result * 255).detach().cpu().numpy()
        result = result.astype(np.uint8)
        return heatmap, result


    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (B, C, H, W)
            class_idx torch.Tensor or list: class indexes for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                  std=[0.229, 0.224, 0.225])
        #
        # # 这两句酌情加
        # transform = transforms.Compose(
        #     [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalize])
        # input = transform(transforms.ToPILImage()(input).convert("RGB")).unsqueeze(0).cuda()
        logit = self.model_arch(input)  # B,#class
        score = torch.gather(logit, 1, class_idx.unsqueeze(1)).squeeze()
        self.model_arch.zero_grad()
        torch.autograd.backward(score,grad_tensors=torch.ones_like(score),retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()
        alpha = gradients.view(b, k, -1).mean(2)
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)
        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)

        for idx, each_map in enumerate(saliency_map):
            map_min, map_max = each_map.min(), each_map.max()
            saliency_map[idx]  = (saliency_map[idx] - map_min).div(map_max - map_min)
        return saliency_map

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)


class GradCAMpp(GradCAM):
    """Calculate GradCAM++ salinecy map.
    A simple example:
        # initialize a model, model_dict and gradcampp
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet.eval()
        model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
        gradcampp = GradCAMpp(model_dict)
        # get an image and normalize with mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        img = load_img()
        normed_img = normalize(img)
        # get a GradCAM saliency map on the class index 10.
        mask, logit = gradcampp(normed_img, class_idx=10)
        # make heatmap from mask and synthesize saliency map using heatmap and img
        heatmap, cam_result = visualize_cam(mask, img)
    Args:
        model_dict (dict): a dictionary that contains 'model_type', 'arch', layer_name', 'input_size'(optional) as keys.
        verbose (bool): whether to print output size of the saliency map givien 'layer_name' and 'input_size' in model_dict.
    """

    def __init__(self, model_dict, verbose=False):
        super(GradCAMpp, self).__init__(model_dict, verbose)

    def forward(self, input, class_idx=None, retain_graph=False):
        """
        Args:
            input: input image with shape of (1, 3, H, W)
            class_idx: class index for calculating GradCAM.
                    If not specified, the class index that makes the highest model prediction score will be used.
        Return:
            mask: saliency map of the same spatial dimension with input
            logit: model output
        """
        b, c, h, w = input.size()

        logit = self.model_arch(input)
        score = torch.gather(logit, 1, class_idx.unsqueeze(1)).squeeze()
        self.model_arch.zero_grad()
        torch.autograd.backward(score,grad_tensors=torch.ones_like(score),retain_graph=retain_graph)
        gradients = self.gradients['value']  # dS/dA
        activations = self.activations['value']  # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                      activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom + 1e-7)
        positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min)

        return saliency_map

def test_grad_cam(folder):

    resnet = torchvision.models.resnet101(pretrained=True)
    resnet.eval()
    resnet.cuda()
    model_dict = dict(model_type='resnet', arch=resnet, layer_name='layer4', input_size=(224, 224))
    cam = GradCAM(model_dict)
    target_folder = "/home1/machen/test_image/"
    shutil.rmtree(target_folder)
    os.makedirs(target_folder,exist_ok=True)
    all_folders = sorted(os.listdir(os.path.dirname(folder)))
    all_inputs = []
    all_class_id = []
    count = 5
    for idx, img_file_name in enumerate(os.listdir(folder)):
        class_id = all_folders.index(folder.split("/")[-1])
        image = cv2.imread(folder+'/'+img_file_name, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32)#H,W,C
        image = np.transpose(image, (2,0,1)) # C,H,W
        image = torch.from_numpy(image) / 255.0
        all_inputs.append(image)
        all_class_id.append(class_id)
        if idx > count:
            break

    all_inputs = torch.stack(all_inputs).cuda()
    all_class_id = torch.from_numpy(np.array(all_class_id)).long().cuda()

    saliency_map = cam.forward(all_inputs, all_class_id)
    # _, heatmap = cam.visualize_cam(saliency_map, image/255)
    # heatmap = np.transpose(heatmap, (1,2,0)) ##CHW->HWC
    # cv2.imwrite(target_folder + "/" + img_file_name, heatmap)
    # print("save to {}".format(target_folder + "/" + img_file_name))
if __name__ == "__main__":
    test_grad_cam("/home1/machen/dataset/ILSVRC2012/validation/n02119022")

