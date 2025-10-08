import cv2 as cv
import numpy as np
import math 

img = cv.imread("images/lena_gray.bmp", cv.IMREAD_GRAYSCALE)

def copier_im(img, r):
    return cv.copyMakeBorder(img, r, r, r, r, borderType=cv.BORDER_CONSTANT, value=0) #on crée un copie de l'image de base pour éviter les problèmes de bord

# défintion d'un patch
def patch(copy_img, pos, off):
    # on utilise copy image pour ne pas avoir de problème de bord
    cx = pos[0] + off
    cy = pos[1] + off # pos c'est les coordonnées du centre du patch
    patch1 = copy_img[cx - off:cx + off + 1, cy - off:cy + off +1] # on fait juste un bloc de pixel de la taille de notre choix
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
    # print(patch1.shape, patch2.shape)
    dist = 0
    for i in range(H):
        for j in range(W):
            dist += (int(patch1[i,j]) - int(patch2[i,j]))**2
    return dist

# def distance(p, q):
#     return np.linalg.norm(p - q)

# fonction de recherche randomisée
def random_search(off,
                  r, # demie taille du patch
                  i, # abscisse de l'offset dans A
                  j, # ordonnée de l'offset dans A
                  H,
                  W,
                  w = 10, 
                  alpha = 1/2):
    u = []
    k = 0
    while w*alpha**k > 1 :
        R = (np.random.uniform(-1, 1), np.random.uniform(-1, 1))   
        u.append((off[0] + w*(alpha**k)*R[0], off[1] + w*(alpha**k)*R[1]))
        k +=1
    u_r = []
    for k in range(len(u)):
        if (r <= i + math.floor(u[k][1]) < H - r) and (r <= j + math.floor(u[k][0]) < W - r): # on vérifie si on est bien dans le patch
            u_r.append((math.floor(u[k][0]), math.floor(u[k][1])))
    return u_r

# propagation
def propag(im1, 
           im2, 
           r, 
           offset,      # l'argument offset est le dictionnaire des offsets
           nb_iters = 4): 
    copy1 = copier_im(im1, r)
    copy2 = copier_im(im2, r)
    H,W = im1.shape
    dist = {}
    # on parcourt d'abord de haut en bas et de gauche à droite, puis on compare avec les voisins de gauche et les voisins du haut
    for _ in range(nb_iters):
        for i in range(H):
            for j in range(W): 
                # (i_prime, j_prime) = (i + offset[(i,j)][1],j + offset[(i,j)][0])
                dist[(i,j)] = distance(patch(copy1, (i,j), r), patch(copy2, (i + offset[(i,j)][1],j + offset[(i,j)][0]), r))
                if (i > 0) and  patch(copy2, (i + offset[(i-1,j)][1],j + offset[(i-1,j)][0]), r).shape == (2*r+1, 2*r+1) :
                    d1 = distance(patch(copy1, (i,j), r), patch(copy2, (i + offset[(i-1,j)][1],j + offset[(i-1,j)][0]), r))
                    if d1 < dist[(i,j)]: 
                        offset[(i,j)] = offset[(i-1,j)]
                        dist[(i,j)] = d1
                if (j > 0) and patch(copy2, (i + offset[(i,j-1)][1],j + offset[(i,j-1)][0]), r).shape == (2*r+1, 2*r+1) :
                    d2 = distance(patch(copy1, (i,j), r), patch(copy2, (i + offset[(i,j-1)][1],j + offset[(i,j-1)][0]), r))
                    if d2 < dist[(i,j)]:
                        offset[(i,j)] = offset[(i,j-1)]
                        dist[(i,j)] = d2

                # recherche aléatoire
                u = random_search(offset[(i,j)], r,i, j, H, W)
                for k in range(len(u)):
                    d_rd = distance(patch(copy1, (i,j), r), patch(copy2, (i + u[k][1],j + u[k][0]), r)) 
                    if d_rd < dist[(i,j)]:
                        offset[(i,j)] = u[k]
                        dist[(i,j)] = d_rd

        # on parcourt ensuite de bas en haut et de droite, puis on compare avec les voisins de droite et du bas
        for i in range(H-1, -1, -1):
            for j in range(W-1, -1, -1): 
                dist[(i,j)] = distance(patch(copy1, (i,j), r), patch(copy2, (i + offset[(i,j)][1],j + offset[(i,j)][0]), r))
                if (i < H-1) and  patch(copy2, (i + offset[(i+1,j)][1],j + offset[(i+1,j)][0]), r).shape == (2*r+1, 2*r+1) :
                    d1 = distance(patch(copy1, (i,j), r), patch(copy2, (i + offset[(i+1,j)][1],j + offset[(i+1,j)][0]), r))
                    if d1 < dist[(i,j)]: 
                        offset[(i,j)] = offset[(i+1,j)]
                        dist[(i,j)] = d1
                if (j < W-1) and patch(copy2, (i + offset[(i,j+1)][1],j + offset[(i,j+1)][0]), r).shape == (2*r+1, 2*r+1) :
                    d2 = distance(patch(copy1, (i,j), r), patch(copy2, (i + offset[(i,j+1)][1],j + offset[(i,j+1)][0]), r))
                    if d2 < dist[(i,j)]:
                        offset[(i,j)] = offset[(i,j+1)]
                        dist[(i,j)] = d2

                # recherche aléatoire
                u = random_search(offset[(i,j)], r, i, j, H, W)
                for k in range(len(u)):
                    d_rd = distance(patch(copy1, (i,j), r), patch(copy2, (i + u[k][1],j + u[k][0]), r)) 
                    if d_rd < dist[(i,j)]:
                        offset[(i,j)] = u[k]
                        dist[(i,j)] = d_rd

    return dist, offset



print(propag(img, img, 4, init(img))[1])