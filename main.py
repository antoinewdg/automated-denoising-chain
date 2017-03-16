import cvu
import time

from numba import cuda as cd

from noise_estimation import estimate_uniform_noise

if __name__ == '__main__':
    t = time.time()
    image = cvu.load_grayscale('files/flowers_1.png')
    noise = estimate_uniform_noise(image)
    print(noise)
    print('Computed in %s' % (time.time() - t))
