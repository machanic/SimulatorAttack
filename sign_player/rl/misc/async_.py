import multiprocessing as mp
import warnings
import numpy as np
from torch import nn
class AbnormalExitWarning(Warning):
    """Warning category for abnormal subprocess exit."""
    pass

def assert_params_not_shared(a, b):
    assert isinstance(a, nn.Module)
    assert isinstance(b, nn.Module)
    a_params = dict(a.named_parameters())
    b_params = dict(b.named_parameters())
    for name, a_param in a_params.items():
        b_param = b_params[name]
        assert a_param.data_ptr != b_param.data_ptr

