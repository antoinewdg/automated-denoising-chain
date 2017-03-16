import glob
import numpy as np
import seaborn as sbn
from matplotlib import pyplot as plt


def display_results(dir_name, title):
    x = None
    names = glob.glob('out/%s/*.txt' % dir_name)
    for n in names:
        clean_name = n.split('/')[-1].split('.')[0]
        x, y = np.loadtxt(n)
        x *= 255
        y *= 255
        plt.plot(x, y, ':', label=clean_name)

    plt.plot(x, x, label='baseline', lw=7)
    plt.xlabel(title)
    plt.legend()
    plt.show()


display_results('uniform_noise', 'Uniform noise without quantization')
display_results('uniform_noise_quantization', 'Uniform noise with quantization')
