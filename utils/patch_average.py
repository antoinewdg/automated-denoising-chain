import math

import numpy as np
import cv2
from numba import jit, float32, vectorize, cuda as cd, float32, int32

import cvu
from utils import get_grid_dims

B_SIZE = 32


@cd.jit(device=True)
def _unflat_coords(a, idx):
    return idx // a.shape[1], idx % a.shape[1]


@cd.jit(device=True)
def inside_bounds(a, i, j):
    return 0 <= i < a.shape[0] and 0 <= j < a.shape[1]


@cd.jit(device=True)
def get_or_zero(a, i, j):
    if not inside_bounds(a, i, j):
        return 0
    return a[i, j]


def make_kernel_patch_average(patch_radius):
    PATCH_RADIUS = patch_radius
    _SH_B_SIZE = 2 * PATCH_RADIUS + B_SIZE

    @cd.jit()
    def _kernel_patch_average(g_in, g_out):
        i, j = cd.grid(2)
        sh_block = cd.shared.array((_SH_B_SIZE, _SH_B_SIZE), float32)
        sh_weights = cd.shared.array((_SH_B_SIZE, _SH_B_SIZE), int32)

        x, y = cd.threadIdx.x, cd.threadIdx.y

        idx = x * B_SIZE + y
        a, b = B_SIZE * cd.blockIdx.x, B_SIZE * cd.blockIdx.y
        # sh_block[0, 0] = 127 * 127
        while idx < _SH_B_SIZE * _SH_B_SIZE:
            u, v = _unflat_coords(sh_block, idx)
            sh_block[u, v] = get_or_zero(g_in, a + u - PATCH_RADIUS, b + v - PATCH_RADIUS)
            sh_weights[u, v] = int32(inside_bounds(g_in, a + u - PATCH_RADIUS, b + v - PATCH_RADIUS))
            idx += B_SIZE * B_SIZE

        cd.syncthreads()
        if not inside_bounds(g_in, i, j):
            return

        acc = 0
        n = 0
        for k in range(2 * PATCH_RADIUS + 1):
            for l in range(2 * PATCH_RADIUS + 1):
                acc += sh_block[x + k, y + l]
                n += sh_weights[x + k, y + l]

        g_out[i, j] = float32(acc / n)

    return _kernel_patch_average


def cuda_patch_average(d_in, d_out, patch_radius):
    g_dims = get_grid_dims(d_in.shape, B_SIZE)
    kernel = make_kernel_patch_average(patch_radius)
    kernel[g_dims, (B_SIZE, B_SIZE)](d_in, d_out)
