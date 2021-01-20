import torch.nn.functional as F
from torch import nn
from collections import defaultdict
import types
from functools import partial, update_wrapper
import torch
import math

def conv_weight_forward(self, x, conv_fc_module_to_name, param_dict):
    module_weight_name = conv_fc_module_to_name[self]["weight"]
    conv_weight = param_dict[module_weight_name]
    conv_bias = self.bias
    if self.bias is not None:
        module_bias_name = conv_fc_module_to_name[self]["bias"]
        conv_bias = param_dict[module_bias_name]
    out = F.conv2d(x, conv_weight, conv_bias, self.stride,
                   self.padding, self.dilation, self.groups)  # B, C, H, W
    return out


def deconv_weight_forward(self, x, conv_fc_module_to_name, param_dict):
    module_weight_name = conv_fc_module_to_name[self]["weight"]
    conv_weight = param_dict[module_weight_name]
    conv_bias = self.bias
    if self.bias is not None:
        module_bias_name = conv_fc_module_to_name[self]["bias"]
        conv_bias = param_dict[module_bias_name]
    if self.padding_mode != 'zeros':
        raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

    output_padding = self._output_padding(x, None, self.stride, self.padding, self.kernel_size)

    return F.conv_transpose2d(
        x, conv_weight, conv_bias, self.stride, self.padding,
        output_padding, self.groups, self.dilation)


def fc_weight_forward(self, x,conv_fc_module_to_name, param_dict):
    module_weight_name = conv_fc_module_to_name[self]["weight"]
    fc_weight = param_dict[module_weight_name]
    fc_bias = self.bias
    if self.bias is not None:
        module_bias_name = conv_fc_module_to_name[self]["bias"]
        fc_bias = param_dict[module_bias_name]
    return F.linear(x, fc_weight, fc_bias)

def bn_forward(self, x, conv_fc_module_to_name, param_dict):
    exponential_average_factor = 0.0
    if self.training and self.track_running_stats:
        self.num_batches_tracked += 1
        if self.momentum is None:  # use cumulative moving average
            exponential_average_factor = 1.0 / self.num_batches_tracked.item()
        else:  # use exponential moving average
            exponential_average_factor = self.momentum
    module_weight_name = conv_fc_module_to_name[self]["weight"]
    weight = param_dict[module_weight_name]
    bias = self.bias
    if self.bias is not None:
        module_bias_name = conv_fc_module_to_name[self]["bias"]
        bias = param_dict[module_bias_name]

    return F.batch_norm(
        x, self.running_mean, self.running_var, weight, bias,
        self.training or not self.track_running_stats,
        exponential_average_factor, self.eps)



class MetaNetwork(nn.Module):
    def __init__(self, network):
        super(MetaNetwork, self).__init__()
        self.network = network
        self.loss_fn = nn.CrossEntropyLoss()
        self.conv_fc_module_to_name = self.construct_module_name_dict()
        self._init_weights()

    def _init_weights(self):
        ''' Set weights to Gaussian, biases to zero '''
        # self.network._initialize_weights()
        torch.manual_seed(1337)
        torch.cuda.manual_seed(1337)
        torch.cuda.manual_seed_all(1337)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                # m.bias.data.zero_() + 1
                if m.bias is not None:
                    m.bias.data = torch.ones(m.bias.data.size())


    def construct_module_name_dict(self):
        module_to_name = defaultdict(dict)
        for name, module in self.network.named_modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d)\
                    or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.BatchNorm3d) \
                    or isinstance(module, nn.BatchNorm1d):
                module_to_name[module]["weight"] = "network.{}.weight".format(name)
                if module.bias is not None:
                    module_to_name[module]["bias"] = "network.{}.bias".format(name)
        return module_to_name

    def backup_orig_forward(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d)\
                or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm1d):
            if not hasattr(module, "orig_forward"):
                f = module.forward
                g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                                       argdefs=f.__defaults__,
                                       closure=f.__closure__)
                g = update_wrapper(g, f)
                module.orig_forward = types.MethodType(g, module)

    def recover_orig_forward(self, module):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.BatchNorm2d) \
                or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.BatchNorm3d)  or isinstance(module, nn.BatchNorm1d):
            module.forward = module.orig_forward


    def replace_forward(self, module, weight):
        if isinstance(module, nn.Conv2d):
            module.forward = partial(types.MethodType(conv_weight_forward, module), conv_fc_module_to_name=self.conv_fc_module_to_name,
                                     param_dict=weight)
        elif isinstance(module, nn.Linear):
            module.forward = partial(types.MethodType(fc_weight_forward, module), conv_fc_module_to_name=self.conv_fc_module_to_name,
                                     param_dict=weight)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module, nn.BatchNorm1d):
            module.forward = partial(types.MethodType(bn_forward, module), conv_fc_module_to_name=self.conv_fc_module_to_name,
                                     param_dict=weight)
        elif isinstance(module, nn.ConvTranspose2d):
            module.forward = partial(types.MethodType(deconv_weight_forward, module),
                                     conv_fc_module_to_name=self.conv_fc_module_to_name,
                                     param_dict=weight)

    def forward(self,x):
        return self.network(x)

    def copy_weights(self, net):
        ''' Set this module's weights to be the same as those of 'net' '''
        for m_from, m_to in zip(net.modules(), self.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d) \
                    or isinstance(m_to, nn.ConvTranspose2d) or isinstance(m_to, nn.BatchNorm3d) or isinstance(m_to, nn.BatchNorm1d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def net_forward(self, x, weight=None):
        self.network.apply(self.backup_orig_forward)  # 备份原本的forward函数
        if weight is not None:
            self.network.apply(partial(self.replace_forward, weight=weight))
        output = self.forward(x)
        self.network.apply(self.recover_orig_forward)
        return output

    def forward_pass(self, in_, target, weight=None):
        input_var = in_.cuda()
        target_var = target.cuda()
        out = self.net_forward(input_var, weight)
        loss = self.loss_fn(out, target_var)
        return loss, out