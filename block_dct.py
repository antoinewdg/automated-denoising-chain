import numpy as np
import cvu
import math
import time

from numba import cuda as cd, float32


@cd.jit(device=True)
def _device_compute_dct_2(block):
    row = cd.local.array(8, float32)
    a = float32(1) / math.sqrt(float32(8))
    b = math.sqrt(float32(2) / 8)

    for i in range(8):
        for k in range(8):
            row[k] = block[i, k]
            block[i, k] = 0

        for k in range(8):
            for n in range(8):
                block[i, k] += row[n] * math.cos((math.pi / 8) * (n + 0.5) * k)

        block[i, 0] *= a
        for k in range(1, 8):
            block[i, k] *= b

    for i in range(8):
        for k in range(8):
            row[k] = block[k, i]
            block[k, i] = 0

        for k in range(8):
            for n in range(8):
                block[k, i] += row[n] * math.cos((math.pi / 8) * (n + 0.5) * k)

        block[0, i] *= a
        for k in range(1, 8):
            block[k, i] *= b


@cd.jit()
def _kernel_compute_dct_blocks(g_image, g_blocks, g_means):
    i, j = cd.grid(2)
    rows, cols = g_image.shape
    if i >= rows - 7 or j >= cols - 7:
        return

    block = cd.local.array((8, 8), float32)

    mean = float32(0)
    for k in range(8):
        for l in range(8):
            v = g_image[i + k, j + l]
            block[k, l] = v
            mean += v

    _device_compute_dct_2(block)

    g_means[i, j] = mean / 64

    for k in range(8):
        for l in range(8):
            b = block[k, l]
            g_blocks[i, j, k, l] = b * b


def compute_dct_blocks(image):
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(image.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(image.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    rows, cols = image.shape

    new_shape = (rows - 7, cols - 7)
    d_image = cd.to_device(image)
    d_means = cd.device_array(new_shape, np.float32)
    d_blocks = cd.device_array((*new_shape, 8, 8), np.float32)
    _kernel_compute_dct_blocks[blockspergrid, threadsperblock](d_image, d_blocks, d_means)

    return d_blocks.copy_to_host(), d_means.copy_to_host()


if __name__ == '__main__':
    image = cvu.load_grayscale('files/flowers_1.png')
    compute_dct_blocks(image)
