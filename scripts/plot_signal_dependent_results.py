import glob
import numpy as np
import seaborn as sbn
from matplotlib import pyplot as plt


def display_results(dir_name, title):
    names = glob.glob('out/%s/*.txt' % dir_name)
    for n in names:
        clean_name = n.split('/')[-1].split('.')[0]
        x, y = np.loadtxt(n)
        x *= 255
        y *= 255
        plt.plot(x, y, ':', label=clean_name)

    x = np.arange(0, 256)
    plt.plot(x, 0.1 * x + (255 * 0.05), label='baseline', lw=7)
    plt.xlabel(title)
    plt.legend()
    plt.show()


display_results('signal_dependent', 'Uniform noise without quantization')
# display_results('uniform_noise_quantization', 'Uniform noise with quantization')
