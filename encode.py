import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def perm_encoding(img, w, h, seed = 0):
    """
        make block permutation encoding on image
        args: img(input image), w(block width), h(block height), seed(random seed)
        Split img into w*h blocks (filled with 0 at edge if not divisable) and shuffle the blocks.
    """
    np.random.seed(seed)
    mb, nb = (img.shape[0] + w - 1) // w, (img.shape[1] + h - 1) // h
    m, n = mb * w, nb * h
    img_pad = np.pad(img, ((0, m - img.shape[0]), (0, n - img.shape[1])))

    # split image into tiles of w*h blocks with shape = ((m * n) / (w * h), w, h)
    tiles = np.array([img_pad[x : x+w, y : y+h] for x in range(0, m, w) for y in range(0, n, h)])
    np.random.shuffle(tiles)
    # merge back to shape = (m, n)
    res = np.block([[np.hstack(tiles[i*nb : (i+1)*nb])] for i in range(mb)])
    return res

def pixel_xor_encoding(img, seed = 0):
    np.random.seed(seed)
    key = np.random.randint(0, 256, size=img.shape, dtype=np.uint8)
    res = img ^ key
    return res

def pixel_add_encoding(img, seed = 0):
    np.random.seed(seed)
    key = np.random.randint(0, 256, size=img.shape, dtype=np.uint8)
    res = (img + key) % 256
    return res

def entropy(img : np.ndarray) -> float:
    h = np.zeros(256)
    m, n = img.shape
    for i in range(m):
        for j in range(n):
            h[img[i][j]] += 1
    res = 0
    for i in range(256):
        if h[i] == 0: continue
        p = h[i] / m / n
        res += -p * np.log2(p)
    return res

if __name__ == "__main__":
    dir = ".\\src\\"
    for name in os.listdir(dir):
        img = cv2.imread(dir + name, 0)
        img_perm = perm_encoding(img, 200, 200)
        plt.subplot(1, 2, 1), plt.imshow(img, 'gray'), plt.title('Original')
        plt.subplot(1, 2, 2), plt.imshow(img_perm, 'gray'), plt.title('permutation encoding')
        plt.show()

        img_xor = pixel_xor_encoding(img)
        img_add = pixel_add_encoding(img)
        plt.subplot(1, 3, 1), plt.imshow(img, 'gray'), plt.title('E={:.4f}'.format(entropy(img))), plt.axis('off')
        plt.subplot(1, 3, 2), plt.imshow(img_xor, 'gray'), plt.title('Xor E={:.4f}'.format(entropy(img_xor))), plt.axis('off')
        plt.subplot(1, 3, 3), plt.imshow(img_add, 'gray'), plt.title('Add E={:.4f}'.format(entropy(img_add))), plt.axis('off')
        plt.show()