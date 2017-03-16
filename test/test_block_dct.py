import numpy as np
import cvu
import cv2
import glob
from scipy import fftpack

from block_dct import compute_dct_blocks


def _slow_dct_blocks(image):
    def dct(a, t=2):
        b = fftpack.dct(a, axis=0, type=t, norm='ortho')
        return fftpack.dct(b, axis=1, type=t, norm='ortho')

    rows, cols = image.shape

    w = 8
    sq_blocks = np.empty((rows - 7, cols - 7, 8, 8), dtype=np.float32)
    means = np.empty((rows - 7, cols - 7), dtype=np.float32)

    for i in range(rows - w + 1):
        for j in range(cols - w + 1):
            # if mask[i, j] == 0:
            #     continu j] == 0:
            #     continue
            block = image[i:i + w, j:j + w]
            means[i, j] = np.mean(block)
            t_block = dct(block)
            sq_blocks[i, j] = np.square(t_block)

    return sq_blocks, means


def test_dct_blocks():
    image = cvu.load_grayscale('files/building.png')
    exp_blocks, exp_means = _slow_dct_blocks(image)
    act_blocks, act_means = compute_dct_blocks(image)

    assert np.all(np.isclose(exp_means, act_means))
    assert np.all(np.isclose(exp_blocks, act_blocks, atol=1e-5))
