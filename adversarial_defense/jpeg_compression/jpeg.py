from advertorch.functional import JPEGEncodingDecoding
try:
    from cStringIO import StringIO as BytesIO
except ImportError:
    from io import BytesIO
import torch
from torchvision import transforms
from PIL import Image

_to_pil_image = transforms.ToPILImage()
_to_tensor = transforms.ToTensor()


class FloatToIntSqueezing(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, max_int, vmin, vmax):
        # here assuming 0 =< x =< 1
        x = (x - vmin) / (vmax - vmin)
        x = torch.round(x * max_int) / max_int
        return x * (vmax - vmin) + vmin

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "backward not implemented", FloatToIntSqueezing)



class JPEGEncodingDecoding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, quality):
        lst_img = []
        for img in x:
            img = _to_pil_image(img.detach().clone().cpu())
            virtualpath = BytesIO()
            img.save(virtualpath, 'JPEG', quality=quality)
            lst_img.append(_to_tensor(Image.open(virtualpath)))
        return torch.stack(lst_img).clone().detach().cuda()

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError(
            "backward not implemented", JPEGEncodingDecoding)

class JPEGFilter(object):
    """
    JPEG Filter.
    :param quality: quality of the output.
    """
    def __init__(self, quality=75):
        super(JPEGFilter, self).__init__()
        self.quality = quality

    def forward(self, x):
        return JPEGEncodingDecoding.apply(x, self.quality)

    def __call__(self, x):
        return self.forward(x)