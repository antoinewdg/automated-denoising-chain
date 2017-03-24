import cv2
import numpy as np
import pytest
from numba import cuda as cd

import cvu

from utils.patch_average import cuda_patch_average


def test_patch_average():
    image = cvu.load_grayscale('files/flowers_2.png')
    expected = cv2.blur(image, (5, 5))

    d_image = cd.to_device(image)
    d_out = cd.device_array(image.shape, image.dtype)
    cuda_patch_average(d_image, d_out, 2)
    actual = d_out.copy_to_host()

    # Except for the borders, we expect the same results
    assert np.all(np.isclose(actual[2:-2, 2:-2], expected[2:-2, 2:-2]))
