import numpy as np
import cvu
import cv2
import glob

from matplotlib import pyplot as plt

from noise_estimation import estimate_uniform_noise, estimate_signal_dependent_noise
from simulated_noise import add_uniform_noise, add_signal_dependent_noise

from consts import *


def test_uniform_noise():
    names = glob.glob('files/*.png')
    for n in names:
        image = cvu.load_grayscale(n)
        stdevs = [0.01 * i for i in range(0, 20, 3)]
        estimations = []
        for stdev in stdevs:
            print("Estimating %s at %s" % (n, stdev))
            noisy = add_uniform_noise(image, stdev)
            estimations.append(estimate_uniform_noise(noisy))

        c = n.split('/')[-1].split('.')[0]
        out = np.array([stdevs, estimations])
        np.savetxt('out/uniform_noise/%s.txt' % c, out)


def test_uniform_noise_with_quantization():
    names = glob.glob('files/*.png')
    for n in names:
        image = cvu.load_grayscale(n)
        stdevs = [0.01 * i for i in range(0, 20, 3)]
        estimations = []
        for stdev in stdevs:
            noisy = add_uniform_noise(image, stdev, quantization=True)
            estimations.append(estimate_uniform_noise(noisy))

        c = n.split('/')[-1].split('.')[0]
        out = np.array([stdevs, estimations])
        np.savetxt('out/uniform_noise_quantization/%s.txt' % c, out)


def test_signal_dependent_noise():
    names = glob.glob('files/*.png')
    for n in names:
        image = cvu.load_grayscale(n)
        noisy = add_signal_dependent_noise(image, 0.1, 0.05)
        out = estimate_signal_dependent_noise(noisy, 7)
        c = n.split('/')[-1].split('.')[0]
        np.savetxt('out/signal_dependent/%s.txt' % c, out)
