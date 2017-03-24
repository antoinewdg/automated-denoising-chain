import cvu
import time

import cv2

from simulated_noise import add_uniform_noise
from denoising.pixelwise import nl_means

if __name__ == '__main__':
    image = cvu.load_color('files/flowers_1.png')
    noisy = add_uniform_noise(image, 10)
    t = time.time()
    new_image = nl_means(noisy, 10)
    print('Computed in %s' % (time.time() - t))
    cv2.imwrite('out.png', new_image)
    cvu.display_blocking(noisy)
    cvu.display_blocking(new_image)
