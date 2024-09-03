import cv2
import numpy as np

img_gry = cv2.imread('/Users/varshanmuhunthan/Desktop/woman_face.jpeg', cv2.COLOR_BGR2GRAY)
img_gry = cv2.medianBlur(img_gry, 3)
edges = cv2.Canny(img_gry,140,180)
edges = cv2.bitwise_not(edges)
print(edges.shape)
cv2.imwrite('/Users/varshanmuhunthan/Desktop/woman_face_edge.jpeg', edges)

edges_3 = np.zeros_like(img_gry)
edges_3[:,:,0] = edges
edges_3[:,:,1] = edges
edges_3[:,:,2] = edges

img_clr = cv2.imread('/Users/varshanmuhunthan/Desktop/woman_face.jpeg', cv2.IMREAD_UNCHANGED)
img_blr = cv2.medianBlur(img_clr, 7)
cv2.imwrite('/Users/varshanmuhunthan/Desktop/woman_face_blur.jpeg', img_blr)
print(img_blr.shape)

# img_blnd = cv2.addWeighted(edges_3, 0.7, img_blr, 1, 0)
img_blnd = np.zeros_like(img_clr)
for h in range(len(img_clr)):
    for w in range(len(img_clr[h])):
        if edges[h,w] == 255 :
            img_blnd[h, w, :] = img_blr[h, w, :]
        else:
            img_blnd[h, w, :] = 0

cv2.imwrite('/Users/varshanmuhunthan/Desktop/woman_face_cartoon.jpeg', img_blnd)

