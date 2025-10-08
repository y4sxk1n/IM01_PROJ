import cv2 as cv
import numpy as np
import math

img = cv.imread("images/lena_gray.bmp", cv.IMREAD_GRAYSCALE)

def copy_im(img, r):
    return cv.copyMakeBorder(img, r, r, r, r, borderType=cv.BORDER_CONSTANT, value=0) #on crée un copie de l'image de base pour éviter les problèmes de bord

# défintion d'un patch
def patch(copy_img, pos, r):
    # on utilise copy image pour ne pas avoir de problème de bord
    cx = pos[0] + r
    cy = pos[1] + r # pos c'est les coordonnées du centre du patch
    patch = copy_img[cx - r:cx + r + 1, cy - r:cy + r +1] # on fait juste un bloc de pixel de la taille de notre choix
    return patch

# initialisation 
def init_off(img):
    H, W = img.shape
    offsets = np.zeros((H, W, 2), dtype=int)
    for i in range(H):
        for j in range(W): 
            offsets[i, j] = (np.random.randint(0, W) - j, np.random.randint(0, H) - i) # position aléatoire dans l'image B
    return offsets

# fonction distance
def distance(patch1, patch2):
    H, W = patch1.shape
    dist = 0
    for i in range(H):
        for j in range(W):
            dist += (patch1[i,j] - patch2[i,j])**2
    return dist

# fonction de recherche randomisée
def random_search(im1, im2, r, offsets, dist, w=10, alpha=0.5):
    ...

# propag
def propag(im1, im2, r, offsets, direction='forward'):
    H, W = im1.shape
    dist = np.zeros((H, W))
    copy1 = copy_im(im1, r)
    copy2 = copy_im(im2, r)

    if direction == 'forward':
        for i in range(H):
            for j in range(W):
                dx, dy = offsets[i, j]
                d_cur = distance(patch(copy1, (i, j), r), patch(copy2, (i + dy, j + dx), r))
                dist[i, j] = d_cur

                # voisin du haut
                if i > 0 and patch(copy2, (i + offsets[i-1, j][1], j + offsets[i-1, j][0]), r).shape == (2*r+1, 2*r+1):
                    d1 = distance(patch(copy1, (i, j), r), patch(copy2, (i + offsets[i-1, j][1], j + offsets[i-1, j][0]), r))
                    if d1 < dist[i, j]:
                        offsets[i, j] = offsets[i-1, j]
                        dist[i, j] = d1

                # voisin de gauche
                if j > 0 and patch(copy2, (i + offsets[i, j-1][1], j + offsets[i, j-1][0]), r).shape == (2*r+1, 2*r+1):
                    d2 = distance(patch(copy1, (i, j), r), patch(copy2, (i + offsets[i, j-1][1], j + offsets[i, j-1][0]), r))
                    if d2 < dist[i, j]:
                        offsets[i, j] = offsets[i, j-1]
                        dist[i, j] = d2

    else:
        for i in range(H-1, -1, -1):
            for j in range(W-1, -1, -1):
                dx, dy = offsets[i, j]
                d_cur = distance(patch(copy1, (i, j), r), patch(copy2, (i + dy, j + dx), r))
                dist[i, j] = d_cur

                # voisin du bas
                if i < H-1 and patch(copy2, (i + offsets[i+1, j][1], j + offsets[i+1, j][0]), r).shape == (2*r+1, 2*r+1):
                    d1 = distance(patch(copy1, (i, j), r), patch(copy2, (i + offsets[i+1, j][1], j + offsets[i+1, j][0]), r))
                    if d1 < dist[i, j]:
                        offsets[i, j] = offsets[i+1, j]
                        dist[i, j] = d1

                # voisin de droite
                if j < W-1 and patch(copy2, (i + offsets[i, j+1][1], j + offsets[i, j+1][0]), r).shape == (2*r+1, 2*r+1):
                    d2 = distance(patch(copy1, (i, j), r),patch(copy2, (i + offsets[i, j+1][1], j + offsets[i, j+1][0]), r))
                    if d2 < dist[i, j]:
                        offsets[i, j] = offsets[i, j+1]
                        dist[i, j] = d2

    return offsets, dist


# patchmatch
def patchmatch(im1, im2, r, offsets, nb_iters = 15): # l'argument offset est le dictionnaire des offsets
    for _ in range(nb_iters):
        # on parcourt d'abord de haut en bas et de gauche à droite, puis on compare avec les voisins de gauche et les voisins du haut
        offsets, dist = propag(im1, im2, r, offsets, direction='forward')
        
        # on parcourt ensuite de bas en haut et de droite, puis on compare avec les voisins de droite et du bas
        offsets, dist = propag(im1, im2, r, offsets, direction='backward')

        # on effectue la recherche randomisée
        # offsets, dist = random_search(im1, im2, r, offsets, dist, w=10, alpha=0.5)

    return offsets, dist

print(patchmatch(img, img, 4, init_off(img))[0])   