import cv2
import numpy as np


def display_blocking(img):
    cv2.imshow("OpenCV", img)
    cv2.waitKey(0)


def load_color(name):
    img = cv2.imread(name)
    return img.astype(np.float32) / 255


def load_grayscale(name):
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    return img.astype(np.float32) / 255
