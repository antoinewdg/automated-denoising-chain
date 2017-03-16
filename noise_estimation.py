import cv2
import cvu
import math
import gc
from scipy import fftpack

from consts import *
from block_dct import compute_dct_blocks


def _compute_bin_sizes(n_elements, n_bins):
    small_size = math.floor(n_elements / n_bins)
    bin_sizes = [small_size for _ in range(n_bins)]
    for i in range(n_elements % n_bins):
        bin_sizes[i] += 1

    return bin_sizes


def find_saturated_blocks(image):
    """
    Create a mask identifying blocks containing saturated pixels.

    This is not done as in the paper, here a pixel is considered saturated
    if its value is too close to zero or too close to 255.

    :param image: grayscale image
    :return:
    """
    mask = 255 * np.ones(image.shape, dtype=np.uint8)
    mask[image <= 1 / 255] = 0
    mask[image >= 254 / 255] = 0
    elt = np.ones((8, 8))
    mask = cv2.erode(mask, elt)
    return mask[:-7, :-7]


def build_dct_blocks(image, mask):
    """
    Build all the overlapping DCT blocks for the image.

    :param image: grayscale image
    :param mask: mask, blocks are not computed where mask is 0
    :return: array of blocks
    """

    def dct(a, t=2):
        b = fftpack.dct(a, axis=0, type=t, norm='ortho')
        return fftpack.dct(b, axis=1, type=t, norm='ortho')

    rows, cols = image.shape
    w = BLOCK_SIZE
    sq_blocks = []
    means = []
    for i in range(rows - w + 1):
        for j in range(cols - w + 1):
            if mask[i, j] == 0:
                continue
            block = image[i:i + w, j:j + w]
            means.append(np.mean(block))
            t_block = dct(block)
            sq_blocks.append(np.square(t_block))

    dct_square_blocks = np.array(sq_blocks)
    dct_means = np.array(means)

    return dct_square_blocks, dct_means


def filter_blocks(square_blocks, means, mask):
    i = 0
    rows, cols = means.shape
    means = means.reshape(rows * cols)
    square_blocks = square_blocks.reshape((rows * cols, 8, 8))
    mask = mask.reshape(rows * cols)

    for j in range(len(mask)):
        if not mask[j]:
            continue
        square_blocks[i] = square_blocks[j]
        means[i] = means[j]
        i += 1

    square_blocks.resize((i, 8, 8))
    means.resize(i)
    # i, j = 0, 0
    # for k in range(square_blocks.shape[0]):
    #     for l in range(square_blocks.shape[1]):
    #         if not mask[i, j]:
    #             continue
    #         square_blocks[i, j] = square_blocks[k, l]
    #         means[i, j] = means[k, l]
    #         i, j = i + 1, j + 1


def estimate_uniform_noise(image):
    """
    Estimate standard deviation of the uniform gaussian noise in the image.

    :param image: grayscale image
    :return:
    """
    mask = find_saturated_blocks(image)
    import time
    t0 = time.time()
    square_blocks, means = compute_dct_blocks(image)
    t1 = time.time()
    square_blocks = square_blocks[mask == 255]
    estimator = DCTNoiseEstimator(square_blocks)
    result = estimator.estimate_noise()
    t2 = time.time()

    print("DCT : %s" % (t1 - t0))
    print("Est : %s" % (t2 - t1))

    return result


def estimate_signal_dependent_noise(image, n_bins):
    mask = find_saturated_blocks(image)
    # cvu.display_blocking(mask)
    # cvu.display_blocking(image)
    sqr_blocks, means = build_dct_blocks(image, mask)
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
    # return np.array([0, 0])


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
        for i, block in enumerate(self.dct_square_blocks):
            v = np.mean(block[LOW_FREQUENCY_MAP])
            self.low_frequency_variances[i] = v

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


# class NoiseEstimator:
#     def __init__(self, image, n_bins):
#         self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         self.dct_square_blocks = []
#         self.dct_means = []
#         self.low_frequency_variances = []
#         self.high_frequency_variances = []
#         self.n_bins = n_bins
#         self.bin_bounds = []
#
#     def get_noise_estimation(self):
#         self._build_dct_blocks()
#         self._compute_low_frequency_variances()
#         self._sort_dct_blocks()
#         self._compute_high_frequency_variances()
#         return self._compute_estimation()
#
#     def _build_dct_blocks(self):
#         rows, cols = self.gray.shape
#         w = BLOCK_SIZE
#         sq_blocks = []
#         means = []
#         for i in range(rows - w):
#             for j in range(cols - w):
#                 block = self.gray[i:i + w, j:j + w]
#                 block = fftpack.dct(block, axis=0, type=2, norm='ortho')
#                 block = fftpack.dct(block, axis=1, type=2, norm='ortho')
#                 sq_blocks.append(np.square(block))
#                 means.append(np.mean(block))
#
#         self.dct_square_blocks = np.array(sq_blocks)
#         self.dct_means = np.array(means)
#
#     # def _build_bins(self):
#     #     bin_bounds = []
#     #     idx = np.argsort(self.dct_means)
#     #     self.dct_means = self.dct_means[idx]
#     #     self.dct_square_blocks = self.dct_square_blocks[idx]
#     #
#     #     for k in range(0, len(self.dct_means), ELEMENTS_PER_BIN):
#     #         lower = self.dct_means[k]
#     #         k_max = min(k + ELEMENTS_PER_BIN - 1, len(self.dct_means) - 1)
#     #         higher = self.dct_means[k_max]
#
#     def _compute_low_frequency_variances(self):
#         variances = []
#         for block in self.dct_square_blocks:
#             variances.append(np.mean(block[LOW_FREQUENCY_MAP]))
#
#         self.low_frequency_variances = np.array(variances)
#
#     def _sort_dct_blocks(self):
#         idx = np.argsort(self.low_frequency_variances)
#         self.dct_square_blocks = self.dct_square_blocks[idx]
#
#     def _compute_high_frequency_variances(self):
#         bin_bounds = []
#         variances = []
#         bin_sizes = _compute_bin_sizes(self.n_bins, len(self.dct_means))
#         i = 0
#         for s in bin_sizes:
#             lower = self.dct_means[i]
#             higher = self.dct_means[i + s - 1]
#             k = int(0.005 * s)
#             bin_bounds.append([lower, higher])
#             variances.append(np.mean(self.dct_square_blocks[i:i + k], axis=0))
#             i += s
#
#         self.high_frequency_variances = np.array(variances)
#         self.bin_bounds = np.array(bin_bounds)
#
#     def _compute_estimation(self):
#         high_frequency_map = np.logical_not(LOW_FREQUENCY_MAP)
#         high_frequency_map[0, 0] = False
#         values = []
#
#         print(self.high_frequency_variances.shape)
#         for variances in self.high_frequency_variances:
#             v = np.sqrt(np.median(variances[high_frequency_map]))
#             values.append(v)
#
#         return np.array(values)


def main():
    image = cvu.load_color("lena_noise.png")
    # cvu.display_blocking(image)
    # print(LOW_FREQUENCY_MAP)
    # sqr, means = co
    # estimator = NoiseEstimator(image, 7)
    sigma = estimate_uniform_noise(image)
    # print(255 * estimator.bin_bounds)
    print(sigma * 255)


if __name__ == '__main__':
    main()
