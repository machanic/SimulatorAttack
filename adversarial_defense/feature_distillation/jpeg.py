import torch
from scipy import fftpack
from PIL import Image
import math
import numpy as np
import argparse
from numba import jit
def load_quantization_table(component, qs=40):
    # Quantization Table for JPEG Standard: https://tools.ietf.org/html/rfc2435
    if component == 'lum':
        q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                      [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56],
                      [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77],
                      [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101],
                      [72, 92, 95, 98, 112, 100, 103, 99]])
    elif component == 'chrom':
        q = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                      [18, 21, 26, 66, 99, 99, 99, 99],
                      [24, 26, 56, 99, 99, 99, 99, 99],
                      [47, 66, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99],
                      [99, 99, 99, 99, 99, 99, 99, 99]])
    elif component == 'dnn':
        q = np.array([[0, 0, 0, 0, 0, 1, 1, 1],
                      [0, 0, 0, 0, 1, 1, 1, 1],
                      [0, 0, 0, 1, 1, 1, 1, 1],
                      [0, 0, 1, 1, 1, 1, 1, 1],
                      [0, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1],
                      [1, 1, 1, 1, 1, 1, 1, 1]])
        q = q * qs + np.ones_like(q)
    return q


def make_table(component, factor, qs=40):
    factor = np.clip(factor, 1, 100)
    if factor < 50:
        q = 5000 / factor
    else:
        q = 200 - factor * 2
    qt = (load_quantization_table(component, qs) * q + 50) / 100
    qt = np.clip(qt, 1, 255)
    return qt


def quantize(block, component, factor=100):
    qt = make_table(component, factor)
    return (block / qt).round()


def dequantize(block, component, factor=100):
    qt = make_table(component, factor)
    return block * qt


def dct2d(block):
    dct_coeff = fftpack.dct(fftpack.dct(block, axis=0, norm='ortho'),
                            axis=1, norm='ortho')
    return dct_coeff


def idct2d(dct_coeff):
    block = fftpack.idct(fftpack.idct(dct_coeff, axis=0, norm='ortho'),
                         axis=1, norm='ortho')
    return block

@jit
def encode(npmat, component, factor):
    rows, cols = npmat.shape[0], npmat.shape[1]
    blocks_count = rows // 8 * cols // 8
    quant_matrix_list = []
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            for k in range(3):
                block = npmat[i:i + 8, j:j + 8, k] - 128.
                dct_matrix = dct2d(block)
                if component == 'jpeg':
                    quant_matrix = quantize(dct_matrix, 'lum' if k == 0 else 'chrom', factor)
                else:
                    quant_matrix = quantize(dct_matrix, component, factor)
                quant_matrix_list.append(quant_matrix)
    return blocks_count, quant_matrix_list

@jit
def decode(blocks_count, quant_matrix_list, component, factor):
    block_side = 8
    image_side = int(math.sqrt(blocks_count)) * block_side
    blocks_per_line = image_side // block_side
    npmat = np.empty((image_side, image_side, 3))
    quant_matrix_index = 0
    for block_index in range(blocks_count):
        i = block_index // blocks_per_line * block_side
        j = block_index % blocks_per_line * block_side
        for c in range(3):
            quant_matrix = quant_matrix_list[quant_matrix_index]
            quant_matrix_index += 1
            if component == 'jpeg':
                dct_matrix = dequantize(quant_matrix, 'lum' if c == 0 else 'chrom', factor)
            else:
                dct_matrix = dequantize(quant_matrix, component, factor)
            block = idct2d(dct_matrix)
            npmat[i:i + 8, j:j + 8, c] = block + 128.
    npmat = np.clip(npmat.round(), 0, 255).astype('uint8')
    return npmat


def dnn_jpeg(image, component='dnn', factor=50):
    return_torch_tensor = False
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
        return_torch_tensor = True
    image = image * 255  # 0-1 --> 0-255
    image_uint8 = image.astype('uint8')
    ycbcr = Image.fromarray(image_uint8, 'RGB').convert('YCbCr')
    npmat = np.array(ycbcr)
    cnt, coeff = encode(npmat, component, factor)
    npmat_decode = decode(cnt, coeff, component, factor)
    image_obj = Image.fromarray(npmat_decode, 'YCbCr').convert('RGB')
    image_array = np.array(image_obj, dtype='float')  / 255.0
    if return_torch_tensor:
        image_array = torch.from_numpy(image_array).cuda()
    return image_array


def convert_images(images, component='dnn', factor=50):
    images = images.permute(0, 2, 3, 1)  # NCHW -> NHWC
    converted_images = []
    for image in images:
        converted_image = dnn_jpeg(image, component, factor)
        converted_images.append(converted_image)
    converted_images = torch.stack(converted_images)
    converted_images = converted_images.permute(0, 3,1,2).float()  # NHWC -> NCHW
    return converted_images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='fig/lena.png', help='image name')
    parser.add_argument('--component', type=str, default='dnn',
                        help='dnn-oriented or jpeg standard')
    parser.add_argument('--factor', type=int, default=50, help='compression factor')
    args = parser.parse_args()

    image = Image.open(args.image)
    image_npmat = np.array(image, dtype='float')
    npmat_jpeg = dnn_jpeg(image_npmat/255.0, component=args.component, factor=args.factor)
    npmat_jpeg = (npmat_jpeg * 255.0).as_type(np.uint8)
    image_obj = Image.fromarray(npmat_jpeg)
    image_obj.save('lena_jpeg.jpg', 'JPEG')


if __name__ == '__main__':
    main()
