import cv2 as cv
import numpy as np

img = cv.imread("images/lena_gray.bmp", cv.IMREAD_GRAYSCALE)

def display_patch(img, x, off):
    cx, cy = x
    patch = img[cx-off:cx+off, cy-off:cy+off]
    return patch

def dist(p, q):
    return np.linalg.norm(p - q)

# shuffled = img.flatten()
# np.random.shuffle(shuffled)
# shuffled = shuffled.reshape(img.shape)

# cv.imshow('Shuffled:', display_patch(img, (0, 0), 100))
# k = cv.waitKey(0)


# print(dist(display_patch(img, (100, 160), 100), display_patch(img, (100, 150), 100)))

def random_search(off,
                  w = 10, 
                  alpha = 1/2):
    u = []
    i = 0
    while w*alpha**i > 1 :
        R = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))
        u.append(off + np.dot(w*(alpha**i),R))
        i +=1
    return u

print(random_search((30, 20)))

print(np.dot(2, (1, 1)))
