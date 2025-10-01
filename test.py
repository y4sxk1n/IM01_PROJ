import cv2 as cv
import numpy as np

img = cv.imread("images/lena_gray.bmp", cv.IMREAD_GRAYSCALE)

def copier_im(img, r):
    return cv.copyMakeBorder(img, r, r, r, r, borderType=cv.BORDER_CONSTANT, value=0) #on crée un copie de l'image de base pour éviter les problèmes de bord

# défintion d'un patch
def patch(copy_img, pos, off = 4):
    # on utilise copy image pour ne pas avoir de problème de bord
    cx = pos[0] + off
    cy = pos[1] + off # pos c'est les coordonnées du centre du patch
    patch1 = copy_img[cx - off:cy + off + 1, cy - off:cy + off +1] # on fait juste un bloc de pixel de la taille de notre choix
    return patch1

# initialisation 
def init(im1):
    H,W=im1.shape 
    offset = {}
    for i in range (H):
        for j in range(W):
            offset[(i,j)] = (np.random.randint(0, W) - j, np.random.randint(0, H) - i) # on tire aléatoirement une postion dans l'image B
    return offset 

# fonction distance
def distance(patch1, patch2):
    H, W = patch1.shape
    dist = 0
    for i in range(H):
        for j in range(W):
            dist += (patch1[i,j] - patch2[i,j])**2
    return dist

# propagation et recherche aléatoire locale
def propag(im1, im2):
    copy1 = copier_im(im1, 4)
    copy2 = copier_im(im2, 4)
    H,W = im1.shape
    dist = {}
    offset = init(im2)
    for j in range(H):
        for i in range(W): 
            d = distance(patch(copy1, (i,j)), patch(copy2, (i + offset[(i,j)][0],j + offset[(i,j)][1])))
            dist[(i,j)] = d
            if d >  dist[(i-1,j)]: # je compare pas la bonne chose je crois il faut que je recalcule avec la distance de ce patch mais avec un offset différent !
                offset[(i,j)] = offset[(i-1,j)]
            elif d > dist[(i,j-1)]:
                offset[(i,j)] = offset[(i,j-1)]
    
            