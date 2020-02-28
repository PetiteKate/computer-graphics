import cv2
import numpy as np


image = cv2.imread("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_1\\images\\1.jpg")
#cv2.imshow("Image", image)

arr = np.array(image)

h,w,bpp = np.shape(image)

print("width: " + str(w))
print("height: " + str(h))
print("bpp: " + str(bpp))


gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray_im)
threshold = cv2.adaptiveThreshold(gray_im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 10)
#retval, threshold = cv2.threshold(gray_im, 2, 255, cv2.THRESH_BINARY)
cv2.imshow('original',image)
cv2.imshow('threshold',threshold)


cv2.waitKey(0)
cv2.destroyAllWindows()