import numpy as np


def add_uniform_noise(image, stdev, quantization=False):
    if stdev == 0:
        return np.copy(image)
    rows, cols = image.shape
    noise = np.random.normal(0, stdev, (rows, cols)).astype(np.float32)
    out = image + noise
    out[out > 255] = 255
    out[out < 0] = 0

    if quantization:
        out = quantize(out)

    return out


def quantize(image):
    return (np.round(image)).astype(np.float32)


def add_signal_dependent_noise(image, a, b, quantization=False):
    rows, cols = image.shape
    noise = np.random.normal(0, 1, (rows, cols)).astype(np.float32)
    sigmas = np.sqrt(b * np.ones((rows, cols), dtype=np.float32) + a * image)
    out = image + (sigmas * noise)
    out[out > 255] = 255
    out[out < 0] = 0

    if quantization:
        out = quantize(out)

    return out
