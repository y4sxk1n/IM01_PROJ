import cv2 as cv
import numpy as np
import math
import matplotlib as plt

img = cv.imread("images/lena_color.tiff", cv.IMREAD_COLOR)

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
    H, W = img.shape[:2]  
    offsets = np.zeros((H, W, 2), dtype=int)
    for i in range(H):
        for j in range(W): 
            offsets[i, j] = (np.random.randint(0, W) - j, np.random.randint(0, H) - i) # position aléatoire dans l'image B
    return offsets

# fonction distance
def distance(p1, p2):
    d = p1.astype(np.float32) - p2.astype(np.float32)
    return float(np.sum(d*d))

# fonction de recherche randomisée
def random_search(im1, im2, r, offsets, dist, w=100, alpha=0.5):
    H, W = im1.shape[:2]  
    w = min(W, H)
    for i in range(H):
        for j in range(W):
            k = 0
            dx, dy = offsets[i, j]
            best_dist = dist[i, j]
            x_best = j + dx
            y_best = i + dy
            p1 = patch(im1, (i, j), r)
            while w * alpha**k > 1 :
                R = (np.random.uniform(-1, 1), np.random.uniform(-1, 1)) 
                # print(x_best + ((w*alpha**k) * R[0]))
                # print(int(x_best + ((w*alpha**k) * R[0])))
                x_new = int(x_best + (w * alpha**k) * R[0])
                y_new = int(y_best + (w * alpha**k) * R[1])
                # if (w * alpha**k) > 5:
                #     print("offsets =", dx, dy,"x_best =", x_best, "scale =", w * alpha**k, "→ x_new =", x_new)


                # print(x_new, y_new)
                if 0 <= x_new < W and 0 <= y_new < H:
                    p2 = patch(im2, (y_new, x_new), r)
                    if p1.shape == p2.shape:
                        d = distance(p1, p2)
                        if d < best_dist:
                            best_dist = d
                            offsets[i, j] = [x_new - j, y_new - i]
                            dist[i, j] = d

                else: 
                    continue    # tant qu'on ne se trouve pas dans l'image la boucle reste au même k
                k+=1

    return offsets, dist

# propag
def propag(im1, im2, r, offsets, direction='forward'):
    H, W = im1.shape[:2]  
    # print(H, W)
    dist = np.zeros((H, W))
    copy1 = copy_im(im1, r)
    copy2 = copy_im(im2, r)
    if direction == 'forward':
        for i in range(H):
            for j in range(W):
                dx, dy = offsets[i, j]
                p1 = patch(copy1, (i, j), r)
                p2 = patch(copy2, (i + dy, j + dx), r)
                if p1.shape[:2] != (2*r + 1, 2*r + 1) or p2.shape[:2] != (2*r + 1, 2*r + 1):
                    continue 
                d_cur = distance(p1, p2)
                dist[i, j] = d_cur

                # voisin du haut
                if i > 0 and patch(copy2, (i + offsets[i-1, j][1], j + offsets[i-1, j][0]), r).shape[:2] == (2*r+1, 2*r+1):
                    d1 = distance(patch(copy1, (i, j), r), patch(copy2, (i + offsets[i-1, j][1], j + offsets[i-1, j][0]), r))
                    if d1 < dist[i, j]:
                        offsets[i, j] = offsets[i-1, j]
                        dist[i, j] = d1

                # voisin de gauche
                if j > 0 and patch(copy2, (i + offsets[i, j-1][1], j + offsets[i, j-1][0]), r).shape[:2] == (2*r+1, 2*r+1):
                    d2 = distance(patch(copy1, (i, j), r), patch(copy2, (i + offsets[i, j-1][1], j + offsets[i, j-1][0]), r))
                    if d2 < dist[i, j]:
                        offsets[i, j] = offsets[i, j-1]
                        dist[i, j] = d2

    else:
        for i in range(H-1, -1, -1):
            for j in range(W-1, -1, -1):
                dx, dy = offsets[i, j]
                p1 = patch(copy1, (i, j), r)
                p2 = patch(copy2, (i + dy, j + dx), r)
                if p1.shape[:2] != (2*r + 1, 2*r + 1) or p2.shape[:2] != (2*r + 1, 2*r + 1):
                    continue 
                d_cur = distance(p1, p2)
                dist[i, j] = d_cur

                # voisin du bas
                if i < H-1 and patch(copy2, (i + offsets[i+1, j][1], j + offsets[i+1, j][0]), r).shape[:2] == (2*r+1, 2*r+1):
                    d1 = distance(patch(copy1, (i, j), r), patch(copy2, (i + offsets[i+1, j][1], j + offsets[i+1, j][0]), r))
                    if d1 < dist[i, j]:
                        offsets[i, j] = offsets[i+1, j]
                        dist[i, j] = d1

                # voisin de droite
                if j < W-1 and patch(copy2, (i + offsets[i, j+1][1], j + offsets[i, j+1][0]), r).shape[:2] == (2*r+1, 2*r+1):
                    d2 = distance(patch(copy1, (i, j), r),patch(copy2, (i + offsets[i, j+1][1], j + offsets[i, j+1][0]), r))
                    if d2 < dist[i, j]:
                        offsets[i, j] = offsets[i, j+1]
                        dist[i, j] = d2

    return offsets, dist


# patchmatch
def patchmatch(im1, im2, r, offsets, nb_iters = 5): # l'argument offset est le dictionnaire des offsets
    for _ in range(1, nb_iters+1):
        print("IT NUMBER: ", _)

        print("FORWARD PROPAG...")

        # on parcourt d'abord de haut en bas et de gauche à droite, puis on compare avec les voisins de gauche et les voisins du haut
        offsets, dist = propag(im1, im2, r, offsets, direction='forward')

        print("END FORWARD PROPAG")

        print("BACKWARD PROPAG...")
        
        # on parcourt ensuite de bas en haut et de droite, puis on compare avec les voisins de droite et du bas
        offsets, dist = propag(im1, im2, r, offsets, direction='backward')

        print("END BACKWARD PROPAG")

        # print("RANDOM SEARCH...")

        # # offsets, dist = random_search(im1, im2, r, offsets, dist, alpha=0.5)

        # print("END RANDOM SEARCH")
    return offsets, dist

# scale = .25

# img1 = cv.imread("images/trainspotting_1.png", cv.IMREAD_GRAYSCALE)
img1 = cv.imread("images/lena_modif.png", cv.IMREAD_COLOR)
# img1 = cv.resize(img1, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
# cv.imshow("image1", img1)
# cv.waitKey(0)
# cv.destroyAllWindows()

# img2 = cv.imread("images/trainspotting_2.png", cv.IMREAD_GRAYSCALE)
# img2 = cv.imread("images/trainspotting_2.png", cv.IMREAD_COLOR)
# img2 = cv.resize(img2, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
# cv.imshow("image2", img2)
# cv.waitKey(0)
# cv.destroyAllWindows()


# offsets = patchmatch(img1, img2, 4, init_off(img1))[0]

offsets = patchmatch(img, img1, 4, init_off(img))[0]
# print(offsets)

h, w = offsets.shape[:2]
for i in range(w):
    for j in range(h):
        if tuple(offsets[i,j]) != (0, 0) :
            print("l'offset à la place", (i,j), "est", offsets[i,j])


def remap(img1, offsets, r):
    img2 = np.zeros_like(img1, dtype=float)
    H, W = img1.shape[:2]  
    weights = np.zeros((H, W), dtype=float)
    C = 1 if img1.ndim == 2 else img1.shape[2] # gère le cas des images en niveaux de gris et en couleur

    for i in range(H):
        for j in range(W):

            dx = offsets[i, j, 0]
            dy = offsets[i, j, 1]
            i2 = i + dy
            j2 = j + dx

            for u in range(-r, r + 1):
                for v in range(-r, r + 1):
                    y = i + u
                    x = j + v
                    y2 = i2 + u
                    x2 = j2 + v

                    if (0 <= y < H and 0 <= x < W and 0 <= y2 < H and 0 <= x2 < W):
                        if C == 1:
                            img2[y2, x2] += img1[y, x]
                        else:
                            img2[y2, x2, :] += img1[y, x, :] # quand c'est une image en couleur il faut gérer les 3 canaux
                        weights[y2, x2] += 1.0

    if C == 1:
        img2 /= np.maximum(weights, 1e-8)
    else:
        img2 /= np.maximum(weights, 1e-8)[:, :, None] # idem pour les images en couleur, il faut gérer les 3 canaux
    img2 = np.clip(img2, 0, 255)
    return img2.astype(np.uint8)

new_img = remap(img, offsets, 4)
cv.imshow("image", new_img)
cv.waitKey(0)
cv.destroyAllWindows()

