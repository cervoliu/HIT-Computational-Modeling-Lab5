"""
    Author : Cervoliu
    add noise to images, showed by matplotlib.pyplot
"""
import os
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

pepper = 0.5
salt = 1 - pepper

def add_pepper_salt_noise(img : np.ndarray, SNR : float = 10, pepper : float = 0.5) -> np.ndarray:
    """
    Args:
        SNR : Signal-to-Noise Rate in percentage. 0 <= SNR <= 100. Defaults to 10.
        pepper : Pepper-noise Rate. Defaults to 0.5.
    """
    w, h = img.shape
    img_new = img.copy()
    for i in range(w):
        for j in range(h):
            if random.random() > 0.01 * SNR: continue
            if random.random() < pepper:
                img_new[i][j] = 0
            else:
                img_new[i][j] = 255
    return img_new

def add_Gaussian_noise(img : np.ndarray, mean : float = 0, variance : float = 0.1) -> np.ndarray:
    image = np.asarray(img / 255.0, dtype=np.float32)
    noise = np.random.normal(mean, variance, img.shape).astype(dtype=np.float32)
    output = image + noise
    output = np.clip(output, 0, 1)
    output = np.uint8(output * 255)
    return output

if __name__ == "__main__":
    dir = ".\\src\\"

    for name in os.listdir(dir):
        img = cv2.imread(dir + name, 0)
        img1 = add_Gaussian_noise(img)
        img2 = add_pepper_salt_noise(img)
        plt.subplot(1, 3, 1), plt.imshow(img, 'gray'), plt.title('Origin')
        plt.subplot(1, 3, 2), plt.imshow(img1, 'gray'), plt.title('Gaussian')
        plt.subplot(1, 3, 3), plt.imshow(img2, 'gray'), plt.title('pepper-salt')
        plt.show()
