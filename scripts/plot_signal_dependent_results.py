import glob
import numpy as np
import seaborn as sbn
from matplotlib import pyplot as plt


def display_results(dir_name, title):
    names = glob.glob('out/%s/*.txt' % dir_name)
    for n in names:
        clean_name = n.split('/')[-1].split('.')[0]

        x = np.arange(0, 256)
        plt.plot(x, np.sqrt(0.3 * x + 5), label='baseline', lw=7)
        x, y = np.loadtxt(n)

        plt.plot(x, y, ':', label=clean_name)

        plt.legend()
        plt.show()


display_results('signal_dependent', 'Signal dependent noise')
