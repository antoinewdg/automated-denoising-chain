import numpy as np
import cvu
import os
from scipy import fftpack

from block_dct import compute_dct_blocks

SLOW_DCT_BLOCKS_FILE = 'test/cached_computation/slow_dct_blocks.npy'
SLOW_DCT_MEANS_FILE = 'test/cached_computation/slow_dct_means.npy'


def _slow_dct_blocks(image):
    if os.path.isfile(SLOW_DCT_BLOCKS_FILE) and os.path.isfile(SLOW_DCT_MEANS_FILE):
        return np.load(SLOW_DCT_BLOCKS_FILE), np.load(SLOW_DCT_MEANS_FILE)

    def dct(a, t=2):
        b = fftpack.dct(a, axis=0, type=t, norm='ortho')
        return fftpack.dct(b, axis=1, type=t, norm='ortho')

    rows, cols = image.shape

    w = 8
    sq_blocks = np.empty((rows - 7, cols - 7, 8, 8), dtype=np.float32)
    means = np.empty((rows - 7, cols - 7), dtype=np.float32)

    for i in range(rows - w + 1):
        for j in range(cols - w + 1):
            block = image[i:i + w, j:j + w]
            means[i, j] = np.mean(block)
            t_block = dct(block)
            sq_blocks[i, j] = np.square(t_block)

    np.save(SLOW_DCT_BLOCKS_FILE, sq_blocks)
    np.save(SLOW_DCT_MEANS_FILE, means)

    return sq_blocks, means


def test_dct_blocks():
    image = cvu.load_grayscale('files/building_1.png')
    exp_blocks, exp_means = _slow_dct_blocks(image)
    act_blocks, act_means = compute_dct_blocks(image)

    assert np.all(np.isclose(exp_means, act_means))
    assert np.all(np.isclose(exp_blocks, act_blocks, atol=255 * 1e-5))
