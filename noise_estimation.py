import cv2
import cvu
import math
import gc
from scipy import fftpack
from numba import jit

from consts import *
from block_dct import compute_dct_blocks


def _compute_bin_sizes(n_elements, n_bins):
    small_size = math.floor(n_elements / n_bins)
    bin_sizes = [small_size for _ in range(n_bins)]
    for i in range(n_elements % n_bins):
        bin_sizes[i] += 1

    return bin_sizes


@jit()
def find_saturated_blocks(image):
    """
    Create a mask identifying blocks containing saturated pixels.

    This is not done as in the paper, here a pixel is considered saturated
    if its value is too close to zero or too close to 255.

    :param image: grayscale image
    :return:
    """
    rows, cols = image.shape
    mask = 255 * np.ones(image.shape, dtype=np.uint8)

    for i in range(rows - 1):
        for j in range(cols - 1):
            if image[i, j] == image[i + 1, j] and image[i, j] == image[i, j + 1] and image[i, j] == image[i + 1, j + 1]:
                mask[i, j] = 0
                mask[i + 1, j] = 0
                mask[i, j + 1] = 0
                mask[i + 1, j + 1] = 0
    # mask[image <= 0.9] = 0
    # mask[image >= 254.1] = 0
    # cvu.display_blocking(mask)
    elt = np.ones((8, 8))
    mask = cv2.erode(mask, elt)
    # cvu.display_blocking(mask)
    return mask[:-7, :-7]


def estimate_uniform_noise(image):
    """
    Estimate standard deviation of the uniform gaussian noise in the image.

    :param image: grayscale image
    :return:
    """
    mask = find_saturated_blocks(image)
    square_blocks, means = compute_dct_blocks(image)
    square_blocks = square_blocks[mask == 255]
    estimator = DCTNoiseEstimator(square_blocks)
    result = estimator.estimate_noise()

    return result


def estimate_signal_dependent_noise(image, n_bins):
    mask = find_saturated_blocks(image)
    sqr_blocks, means = compute_dct_blocks(image)
    sqr_blocks = sqr_blocks[mask == 255]
    means = means[mask == 255]
    bin_sizes = _compute_bin_sizes(len(means), n_bins)

    idx = np.argsort(means)
    sqr_blocks = sqr_blocks[idx]
    means = means[idx]

    i = 0
    y = []
    x = []

    for s in bin_sizes:
        estimator = DCTNoiseEstimator(sqr_blocks[i:i + s])
        y.append(estimator.estimate_noise())
        x.append(np.mean(means[i:i + s]))
        i += s

    return np.array([x, y])


@jit()
def _compute_low_frequency_variance(dct_square_blocks, out, lf):
    for i in range(len(dct_square_blocks)):
        acc = 0
        count = 0
        for k in range(8):
            for l in range(8):
                acc += int(lf[k, l]) * dct_square_blocks[i, k, l]
                count += int(lf[k, l])
        out[i] = acc / count


class DCTNoiseEstimator:
    def __init__(self, dct_square_blocks):
        l = len(dct_square_blocks)
        self.dct_square_blocks = dct_square_blocks
        self.low_frequency_variances = np.zeros(l, dtype=float)
        self.high_frequency_block = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=float)

    def estimate_noise(self):
        """
        Perform noise estimation from the square values of the DCT blocks.
        :return: noise standard deviation
        """
        self._compute_low_frequency_variances()
        self._sort_dct_blocks()
        self._compute_high_frequency_block()
        return self._compute_estimation()

    def _compute_low_frequency_variances(self):
        """
        Compute the low frequency variance for each block.

        Section 1.2.3 of the paper.
        """
        _compute_low_frequency_variance(self.dct_square_blocks,
                                        self.low_frequency_variances,
                                        LOW_FREQUENCY_MAP)

    def _sort_dct_blocks(self):
        """
        Sort the DCT blocks according to the low frequency variance.

        Beginning of 1.2.4 in the paper.
        """
        idx = np.argsort(self.low_frequency_variances)
        self.dct_square_blocks = self.dct_square_blocks[idx]

    def _compute_high_frequency_block(self):
        """
        Compute the block representing the high frequencies.

        End of 1.2.4 in the paper.
        """
        k = int(0.005 * len(self.dct_square_blocks))
        self.high_frequency_block = np.mean(self.dct_square_blocks[:k], axis=0)

    def _compute_estimation(self):
        """
        Final computation of the standard deviation.

        Section 1.2.5 in the paper.
        :return:
        """
        high_frequency_map = np.logical_not(LOW_FREQUENCY_MAP)
        high_frequency_map[0, 0] = False
        return np.sqrt(np.median(self.high_frequency_block[high_frequency_map]))
