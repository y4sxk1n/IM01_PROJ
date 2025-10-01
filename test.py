import cv2 as cv
import numpy as np

img = cv.imread("images/lena_gray.bmp", cv.IMREAD_GRAYSCALE)

# défintion d'un patch
def patch(img, pos, off):
    cx, cy = pos # pos c'est les coordonnées du centre du patch
    patch = np.zeros(2*off + 1, 2*off + 1) #on initialise à une matrice de zero pour gérer les problèmes de bords
    for i in range(2*(off+1)):
        for j in range(2*(off+1)):
            patch = img[cx - off:cy + off, cy - off:cy + off] # on fait juste un bloc de pixel de la taille de notre choix
    return patch1

# initialisation 
def init(im1):
    n=im1.shape 
    offset = {}
    for i in range (n[0]):
        for j in range(n[1]):
            offset[(i,j)] = (np.random.randint(0, n[1]) - j, np.random.randint(0, n[0]) - i)
    return offset 

# propagation et recherche aléatoire locale
def propag(im1, im2):
    dist = {}
