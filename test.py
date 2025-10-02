import cv2 as cv
import numpy as np

img = cv.imread("images/lena_gray.bmp", cv.IMREAD_GRAYSCALE)

def copier_im(img, r):
    return cv.copyMakeBorder(img, r, r, r, r, borderType=cv.BORDER_CONSTANT, value=0) #on crée un copie de l'image de base pour éviter les problèmes de bord

# défintion d'un patch
def patch(copy_img, pos, off):
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

# fonction de recherche randomisée
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

# propagation
def propag(im1, 
           im2, 
           r, 
           offset ): # l'argument offset est le dictionnaire des offsets
    copy1 = copier_im(im1, r)
    copy2 = copier_im(im2, r)
    H,W = im1.shape
    dist = {}
    # on parcourt d'abord de haut en bas et de gauche à droite, puis on compare avec les voisins de gauche et les voisins du haut
    for j in range(H):
        for i in range(W): 
            dist[(i,j)] = distance(patch(copy1, (i,j), r), patch(copy2, (i + offset[(i,j)][1],j + offset[(i,j)][0]), r))
            if i > 0 :
                d1 = distance(patch(copy1, (i,j), r), patch(copy2, (i + offset[(i-1,j)][1],j + offset[(i-1,j)][0]), r))
                if d1 < dist[(i,j)]: 
                    offset[(i,j)] = offset[(i-1,j)]
                    dist[(i,j)] = d1
            if j > 0:
                d2 = distance(patch(copy1, (i,j), r), patch(copy2, (i + offset[(i,j-1)][1],j + offset[(i,j-1)][0]), r))
                if d2 < dist[(i,j)]:
                    offset[(i,j)] = offset[(i,j-1)]
                    dist[(i,j)] = d2

            # recherche aléatoire
            u = random_search(offset[(i,j)])
            for k in range(len(u)):
                d_rd = distance(patch(copy1, (i,j), r), patch(copy2, (i + u[k][1],j + u[k][0]), r)) < dist[(i,j)]
                if d_rd < dist[(i,j)]:
                    offset[(i,j)] = u[k]
                    dist[(i,j)] = d_rd

    # on parcourt ensuite de bas en haut et de droite, puis on compare avec les voisins de droite et du bas
    for j in range(H-1, -1, -1):
        for i in range(W-1, -1, -1): 
            dist[(i,j)] = distance(patch(copy1, (i,j), r), patch(copy2, (i + offset[(i,j)][1],j + offset[(i,j)][0]), r))
            if i < H-1 :
                d1 = distance(patch(copy1, (i,j), r), patch(copy2, (i + offset[(i+1,j)][1],j + offset[(i+1,j)][0]), r))
                if d1 < dist[(i,j)]: 
                    offset[(i,j)] = offset[(i+1,j)]
                    dist[(i,j)] = d1
            if j < W-1:
                d2 = distance(patch(copy1, (i,j), r), patch(copy2, (i + offset[(i,j+1)][1],j + offset[(i,j+1)][0]), r))
                if d2 < dist[(i,j)]:
                    offset[(i,j)] = offset[(i,j+1)]
                    dist[(i,j)] = d2

            # recherche aléatoire
            u = random_search(offset[(i,j)])
            for k in range(len(u)):
                d_rd = distance(patch(copy1, (i,j), r), patch(copy2, (i + u[k][1],j + u[k][0]), r)) < dist[(i,j)]
                if d_rd < dist[(i,j)]:
                    offset[(i,j)] = u[k]
                    dist[(i,j)] = d_rd

    return dist, offset

