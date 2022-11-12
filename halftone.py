import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from makenoise import add_Gaussian_noise
from makenoise import add_pepper_salt_noise

def Error_Diffusion(x : np.ndarray, thr : int = 127) -> np.ndarray:
    m, n = x.shape
    v = np.array(x)
    b = np.zeros((m, n))
    k = np.array([
    [0, 0, 0, 7, 5],
    [3, 5, 7, 5, 3],
    [1, 3, 5, 3, 1]
    ]) / 48
    for i in range(m):
        for j in range(n):
            if v[i][j] < thr:
                b[i][j] = 0
            else:
                b[i][j] = 255
            e = v[i][j] - b[i][j]
            for di in range(3):
                for dj in range(5):
                    if i + di < m and 0 <= j - 2 + dj < n:
                        v[i + di][j - 2 + dj] += e * k[di][dj]
    return b

if __name__ == "__main__":
    dir = ".\\src\\"
    for name in os.listdir(dir):
        img = cv2.imread(dir + name, 0)
        img_ed = Error_Diffusion(img)
        img_noisy = add_Gaussian_noise(img)
        img_noisy = add_pepper_salt_noise(img_noisy)
        img_noisy_ed = Error_Diffusion(img_noisy)
        plt.subplot(2, 2, 1), plt.imshow(img, 'gray'), plt.title('original'), plt.axis('off')
        plt.subplot(2, 2, 2), plt.imshow(img_ed, 'gray'), plt.title('halftone of original'), plt.axis('off')
        plt.subplot(2, 2, 3), plt.imshow(img_noisy, 'gray'), plt.title('noisy'), plt.axis('off')
        plt.subplot(2, 2, 4), plt.imshow(img_noisy_ed, 'gray'), plt.title('halftone of noisy'), plt.axis('off')
        plt.show()
