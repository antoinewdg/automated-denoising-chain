import math

import numpy as np
import cv2
from numba import vectorize, cuda as cd

from .params import get_params_color
from utils import get_grid_dims, call_kernel
from utils.patch_average import make_kernel_patch_average


@cd.jit(device=True, inline=True)
def _get_im_coord(i, di):
    if di >= 0:
        return i
    else:
        return i - di


@cd.jit(device=True, inline=True)
def _get_im_reverse_coord(i, di):
    if di >= 0:
        return i + di
    else:
        return i


B_SIZE = 32


@cd.jit()
def _kernel_square_distances(g_image, di, dj, g_sqr_distances):
    i, j = cd.grid(2)
    if i >= g_sqr_distances.shape[0] or j >= g_sqr_distances.shape[1]:
        return
    a, b = _get_im_coord(i, di), _get_im_coord(j, dj)

    d2 = 0
    for k in range(3):
        diff = g_image[a, b, k] - g_image[a + di, b + dj, k]
        d2 += diff * diff

    g_sqr_distances[i, j] = d2 / 3


@cd.jit('void(float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], int32, int32)')
def _kernel_add_weighted_values(g_image, g_accumulator, g_normalizers, g_weights, di, dj):
    i, j = cd.grid(2)
    if i >= g_weights.shape[0] or j >= g_weights.shape[1]:
        return
    a, b = _get_im_coord(i, di), _get_im_coord(j, dj)
    x, y = _get_im_reverse_coord(i, di), _get_im_reverse_coord(j, dj)

    weight = g_weights[i, j]
    for k in range(3):
        g_accumulator[x, y, k] += weight * g_image[a, b, k]
    g_normalizers[x, y] += weight


def compute_sqr_distances(d_image, di, dj, d_sqr_distances):
    g_dims = get_grid_dims(d_sqr_distances.shape, B_SIZE)
    _kernel_square_distances[g_dims, (B_SIZE, B_SIZE)](d_image, di, dj, d_sqr_distances)


def _make_exp_ufunc(sigma2, h2):
    @vectorize(['float32(float32)'], target='cuda')
    def exp_ufunc(d2):
        return 255 * math.exp(-max(0, d2 - sigma2) / h2)

    return exp_ufunc


class CudaNLMeans:
    def __init__(self, sigma):
        self.patch_radius, self.window_radius, h = get_params_color(sigma)
        self.h2 = h * h
        self.sigma2 = sigma * sigma
        self._kernel_patch_average = make_kernel_patch_average(self.patch_radius)
        self._exp_unfunc = _make_exp_ufunc(self.sigma2, self.h2)

    def compute_offset_weights(self, d_image, di, dj):
        rows, cols = d_image.shape[:2]
        d_sqr_distances = cd.device_array((rows - abs(di), cols - abs(dj)), np.float32)
        d_patch_distances = cd.device_array(d_sqr_distances.shape, np.float32)
        compute_sqr_distances(d_image, di, dj, d_sqr_distances)
        call_kernel[self._kernel_patch_average, d_sqr_distances, B_SIZE](d_sqr_distances, d_patch_distances)
        return self._exp_unfunc(d_patch_distances)

    def add_weighted_values(self, d_image, d_accumulators, d_normalizers, d_weights, di, dj):
        call_kernel[_kernel_add_weighted_values, d_weights, B_SIZE](
            d_image, d_accumulators, d_normalizers, d_weights, di, dj
        )

    def compute_nl_means(self, image):
        d_image = cd.to_device(image)
        rows, cols = d_image.shape[:2]
        d_normalizers = cd.to_device(np.ones((rows, cols), np.float32))
        d_accumulators = cd.to_device(image)

        for di in range(0, self.window_radius + 1):
            for dj in range(-self.window_radius, self.window_radius + 1):
                if di == 0 and dj == 0:
                    continue
                d_weights = self.compute_offset_weights(d_image, di, dj)
                self.add_weighted_values(d_image, d_accumulators, d_normalizers, d_weights, di, dj)

        n = d_normalizers.copy_to_host()
        return d_accumulators.copy_to_host() / np.dstack((n, n, n))


def nl_means(image, sigma):
    c = CudaNLMeans(sigma)
    return c.compute_nl_means(image)
