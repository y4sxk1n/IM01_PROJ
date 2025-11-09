import cv2 as cv
import numpy as np
import math
import matplotlib as plt

def copy_im(img, r):
    return cv.copyMakeBorder(img, r, r, r, r, borderType=cv.BORDER_CONSTANT, value=0) #on crée un copie de l'image de base pour éviter les problèmes de bord

# défintion d'un patch
def patch(copy_img, pos, r):
    # on utilise copy image pour ne pas avoir de problème de bord
    cx = pos[0]
    cy = pos[1]  # pos c'est les coordonnées du centre du patch
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

R_forbidden = 4
R2 = R_forbidden * R_forbidden

# fonction pour vérifier si un offset est dans la zone interdite
def is_forbidden_offset(dx, dy, R2):
    return (dx * dx + dy * dy) < R2

# fonction distance
def distance(p1, p2):
    d = p1.astype(np.float32) - p2.astype(np.float32)
    return float(np.sum(d*d))

# fonction de recherche randomisée
def random_search(im1, im2, r, offsets, dist, w=20, alpha=0.5):
    H, W = im1.shape[:2]  
    copy1 = copy_im(im1, r)
    copy2 = copy_im(im2, r)
    for i in range(H):
        for j in range(W):
            k = 0
            dx, dy = offsets[i, j]
            best_dist = dist[i, j]
            if best_dist == 0 or not np.isfinite(best_dist):
                best_dist = np.inf
            x_best = j + dx
            y_best = i + dy
            p1 = patch(copy1, (i + r, j + r), r)
            while w * alpha**k > 1 :
                R = (np.random.uniform(-1, 1), np.random.uniform(-1, 1)) 
                x_new = int(x_best + (w * alpha**k) * R[0])
                y_new = int(y_best + (w * alpha**k) * R[1])

                if not (0 <= x_new < W and 0 <= y_new < H):
                    continue

                if is_forbidden_offset(x_new - j, y_new - i, R2):
                    continue

                p2 = patch(copy2, (y_new + r, x_new + r), r)
                if p1.shape == p2.shape:
                    d = distance(p1, p2)
                    if d < best_dist:
                        best_dist = d
                        offsets[i, j] = [x_new - j, y_new - i]
                        dist[i, j] = d
                        x_best, y_best = x_new, y_new
                k += 1
    
    return offsets, dist

# propag
def propag(im1, im2, r, offsets, direction='forward'):
    H, W = im1.shape[:2]  
    dist = np.full((H, W), np.inf, dtype=np.float32)
    copy1 = copy_im(im1, r)
    copy2 = copy_im(im2, r)
    if direction == 'forward':
        for i in range(H):
            for j in range(W):
                dx, dy = offsets[i, j]
                
                p1 = patch(copy1, (i + r, j + r), r)

                # verif offset interdit
                if not is_forbidden_offset(dx, dy, R2):
                    p2 = patch(copy2, (i + dy + r, j + dx + r), r)
                    if p1.shape[:2] == (2*r + 1, 2*r + 1) and p2.shape[:2] == (2*r + 1, 2*r + 1):
                        d_cur = distance(p1, p2)
                        dist[i, j] = d_cur

                # voisin du haut
                if i > 0 and patch(copy2, (i + offsets[i-1, j][1] + r, j + offsets[i-1, j][0] + r), r).shape[:2] == (2*r+1, 2*r+1):
                    cand_dx, cand_dy = offsets[i-1, j]
                    if not is_forbidden_offset(cand_dx, cand_dy, R2):
                        d1 = distance(patch(copy1, (i + r, j + r), r), patch(copy2, (i + offsets[i-1, j][1] + r, j + offsets[i-1, j][0] + r), r))
                        if d1 < dist[i, j]:
                            offsets[i, j] = offsets[i-1, j]
                            dist[i, j] = d1

                # voisin de gauche
                if j > 0 and patch(copy2, (i + offsets[i, j-1][1] + r, j + offsets[i, j-1][0] + r), r).shape[:2] == (2*r+1, 2*r+1):
                    cand_dx, cand_dy = offsets[i, j-1]
                    if not is_forbidden_offset(cand_dx, cand_dy, R2):    
                        d2 = distance(patch(copy1, (i + r, j + r), r), patch(copy2, (i + offsets[i, j-1][1] + r, j + offsets[i, j-1][0] + r), r))
                        if d2 < dist[i, j]:
                            offsets[i, j] = offsets[i, j-1]
                            dist[i, j] = d2

    else:
        for i in range(H-1, -1, -1):
            for j in range(W-1, -1, -1):
                dx, dy = offsets[i, j]
                
                p1 = patch(copy1, (i + r, j + r), r)

                # verif offset interdit
                if not is_forbidden_offset(dx, dy, R2):
                    p2 = patch(copy2, (i + dy + r, j + dx + r), r)
                    if p1.shape[:2] == (2*r + 1, 2*r + 1) and p2.shape[:2] == (2*r + 1, 2*r + 1):
                        d_cur = distance(p1, p2)
                        dist[i, j] = d_cur

                # voisin du bas
                if i < H-1 and patch(copy2, (i + offsets[i+1, j][1] + r, j + offsets[i+1, j][0] + r), r).shape[:2] == (2*r+1, 2*r+1):
                    cand_dx, cand_dy = offsets[i+1, j]
                    if not is_forbidden_offset(cand_dx, cand_dy, R2):
                        d1 = distance(patch(copy1, (i + r, j + r), r), patch(copy2, (i + offsets[i+1, j][1] + r, j + offsets[i+1, j][0] + r), r))
                        if d1 < dist[i, j]:
                            offsets[i, j] = offsets[i+1, j]
                            dist[i, j] = d1

                # voisin de droite
                if j < W-1 and patch(copy2, (i + offsets[i, j+1][1] + r, j + offsets[i, j+1][0] + r), r).shape[:2] == (2*r+1, 2*r+1):
                    cand_dx, cand_dy = offsets[i, j+1]
                    if not is_forbidden_offset(cand_dx, cand_dy, R2):
                        d2 = distance(patch(copy1, (i + r, j + r), r), patch(copy2, (i + offsets[i, j+1][1] + r, j + offsets[i, j+1][0] + r), r))
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

        print("RANDOM SEARCH...")

        offsets, dist = random_search(im1, im2, r, offsets, dist, alpha=0.5)

        print("END RANDOM SEARCH")

    return offsets, dist

# reconstruction de l'image 2 à partir de l'image 1 et des offsets
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

import matplotlib.pyplot as plt

# visualisation des offsets
def make_offset_bgr_from_field(offsets):
    dx = offsets[..., 0].astype(np.float32)
    dy = offsets[..., 1].astype(np.float32)

    def norm01(a):
        amin = a.min()
        amax = a.max()
        if amax == amin:
            return np.zeros_like(a, dtype=np.float32)
        return (a - amin) / (amax - amin)

    dx_n = norm01(dx)
    dy_n = norm01(dy)
    mix = 1.0 - 0.5 * (dx_n + dy_n)

    H, W = dx.shape
    img = np.zeros((H, W, 3), dtype=np.float32)
    img[..., 0] = mix
    img[..., 1] = dy_n
    img[..., 2] = dx_n

    return (img * 255.0).clip(0, 255).astype(np.uint8)



scale = .5

img1 = cv.imread("images/lena_modif_2.png", cv.IMREAD_COLOR)
# img1 = cv.imread("images/lena_modif.png", cv.IMREAD_COLOR)
# img1 = cv.imread("images/cure_fr_modif.png", cv.IMREAD_COLOR)


# img1 = cv.imread("images/trainspotting_1.png", cv.IMREAD_GRAYSCALE)
# img2 = cv.imread("images/trainspotting_2.png", cv.IMREAD_GRAYSCALE)
# img1 = cv.imread("images/trainspotting_1.png", cv.IMREAD_COLOR)
# img2 = cv.imread("images/trainspotting_2.png", cv.IMREAD_COLOR)
# img2 = cv.resize(img2, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
# img1 = cv.resize(img1, None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)
# cv.imshow("image2", img2)
# cv.waitKey(0)
# cv.destroyAllWindows()


# offsets = patchmatch(img1, img2, 4, init_off(img1))[0]
offsets = patchmatch(img1, img1, 4, init_off(img1), nb_iters=5)[0]
# offsets = init_off(img1)


offsetsmap = make_offset_bgr_from_field(offsets)
plt.imshow(cv.cvtColor(offsetsmap, cv.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# new_img = remap(img, offsets, 3)
# cv.imshow("image", new_img)
# cv.waitKey(0)
# cv.destroyAllWindows()


