import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_7\\images\\1.jpg")

laplacian1 = cv2.Laplacian(img,cv2.CV_8U)
laplacian2 = cv2.Laplacian(img,cv2.CV_16U)
laplacian3 = cv2.Laplacian(img,cv2.CV_16S)
laplacian4 = cv2.Laplacian(img,cv2.CV_32F)
laplacian5 = cv2.Laplacian(img,cv2.CV_64F)


titles = ['Original Image', 'laplacian 1', 'laplacian 2', 'laplacian 3', 'laplacian 4', 'laplacian 5']
images = [img, laplacian1, laplacian2, laplacian3, laplacian4, laplacian5]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)

vis = np.concatenate((sobelx, sobely), axis=1)
cv2.imshow("All", vis)

kernel_sharpening = np.array([[-1,-1,-1],
                              [-1, 9,-1],
                              [-1,-1,-1]])

sharpened_img = cv2.filter2D(img, -1, kernel_sharpening)
median = cv2.medianBlur(img,5)
vis = np.concatenate((sharpened_img, median), axis=1)
cv2.imshow("All", vis)

_, threshold = cv2.threshold(img, 125, 255, cv2.THRESH_BINARY)
cv2.imshow('', threshold)

titles = ['Original Image', 'laplacian 1', 'laplacian 2', 'laplacian 3', 'laplacian 4', 'laplacian 5']
images = [img, sobelx, sobely, sharpened_img, median, threshold]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
plt.xticks([]),plt.yticks([])#plt.show()


cv2.waitKey(0)
cv2.destroyAllWindows()