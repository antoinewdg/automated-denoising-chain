import cvu
import numpy as np
import pytest
import itertools

from simulated_noise import (add_uniform_noise, add_signal_dependent_noise,
                             quantize)


class TestUniformNoise:
    def test_zero_stdev(self):
        image = cvu.load_grayscale('files/woman.png')
        noisy = add_uniform_noise(image, 0)
        assert np.all(image == noisy)

    def test_values(self):
        image = cvu.load_grayscale('files/uniform.png')
        stdevs = [0.05, 0.1, 0.15, 0.2]
        for stdev in stdevs:
            noisy = add_uniform_noise(image, stdev)
            actual = np.std(noisy)
            assert actual == pytest.approx(stdev, rel=0.05)


class TestSignalDependentNoise:
    def test_signal_dependent(self):
        image = np.zeros((500, 500))
        image[0:250, 0:250] = 2 / 7
        image[0:250, 250:500] = 3 / 7
        image[250:500, 0:250] = 4 / 7
        image[250:500, 250:500] = 5 / 7
        idx = [(0, 0), (0, 250), (250, 0), (250, 250)]

        b_values = [0, 0.03, 0.05]
        a_values = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

        for a, b in itertools.product(a_values, b_values):
            noisy = add_signal_dependent_noise(image, a, b)
            for i, j in idx:
                v = image[i, j]
                sub = noisy[i:i + 250, j:j + 250]
                assert np.mean(sub) == pytest.approx(v, rel=0.05)
                assert np.std(sub) == pytest.approx(a * v + b, rel=0.05)


def test_quantization_error():
    image = np.random.uniform(0, 1, (500, 500))
    quantized = quantize(image)

    variance = np.sqrt(np.mean(np.square(image - quantized)))
    assert variance == pytest.approx(1 / (np.sqrt(12) * 255), rel=0.01)
