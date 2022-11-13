import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from makenoise import add_Gaussian_noise
from makenoise import add_pepper_salt_noise

def OTSU(img : np.ndarray) -> np.ndarray:
        h = np.zeros(256, dtype=np.uint)
        m, n = img.shape
        for i in range(m):
            for j in range(n):
                h[img[i][j]] += 1

        threshold = 0
        max_var = 0
        for thr in range(256):
            p0 = 0
            p1 = 0
            u0 = 0
            u1 = 0
            for i in range(256):
                if i <= thr:
                    p0 += h[i]
                    u0 += i * h[i]
                else:
                    p1 += h[i]
                    u1 += i * h[i]
            if p0 > 0: u0 /= p0
            if p1 > 0: u1 /= p1
            p0 /= m * n
            p1 /= m * n
            inter_var = p0 * p1 * ((u0 - u1) ** 2)
            if inter_var > max_var:
                max_var = inter_var
                threshold = thr
        res = np.zeros(img.shape)
        for i in range(m):
            for j in range(n):
                if img[i][j] > threshold:
                    res[i][j] = 1
                else:
                    res[i][j] = 0
        return res


if __name__ == "__main__":
    dir = ".\\src\\"
    for name in os.listdir(dir):
        img = cv2.imread(dir + name, 0)
        img_otsu = OTSU(img)
        img_noisy = add_Gaussian_noise(img)
        img_noisy = add_pepper_salt_noise(img_noisy)
        img_noisy_otsu = OTSU(img_noisy)
        plt.subplot(2, 2, 1), plt.imshow(img, 'gray'), plt.title('original'), plt.axis('off')
        plt.subplot(2, 2, 2), plt.imshow(img_otsu, 'gray'), plt.title('OTSU of original'), plt.axis('off')
        plt.subplot(2, 2, 3), plt.imshow(img_noisy, 'gray'), plt.title('noisy'), plt.axis('off')
        plt.subplot(2, 2 ,4), plt.imshow(img_noisy_otsu, 'gray'), plt.title('OTSU of noisy'), plt.axis('off')
        plt.show()
