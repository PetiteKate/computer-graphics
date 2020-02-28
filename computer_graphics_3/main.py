import cv2
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt

original = cv2.imread("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_3\\images\\3.jpg")
#cv2.imshow("Original Image", original)

inversion = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
#cv2.imshow('Inversion', inversion)

gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
ret, threshold_image = cv2.threshold(gray_image, 127, 255, 0)
#cv2.imshow("Black and white image", threshold_image)

(h, w, d) = original.shape
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 90, 1.0)
rotated = cv2.warpAffine(original, M, (w, h))
#cv2.imshow("Rotate 90 degrees", rotated)

cropped = original[100:500, 100:2000]
#cv2.imshow("Framing", cropped)

ret, threshold = cv2.threshold(original, 127, 255, 0)
#cv2.imshow("Zeroing pixels", threshold)

#vis = np.concatenate((original, inversion), axis=1)
#cv2.imshow("All", vis)

titles = ['Original Image', 'Inversion', 'Black and white', 'Rotate 90 degrees', 'Framing', 'Zeroing pixels']
images = [original, inversion, threshold_image, rotated, cropped, threshold]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

#cv2.imwrite("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_3\\im\\OriginalImage.jpg", original)
#cv2.imwrite("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_3\\im\\Inversion.jpg", inversion)
#cv2.imwrite("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_3\\im\\BlackandWhite.jpg", threshold_image)
#cv2.imwrite("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_3\\im\\Rotate 90 degrees.jpg", rotated)
#cv2.imwrite("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_3\\im\\Framing.jpg", cropped)
#cv2.imwrite("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_3\\im\\ZeroingPixels.jpg", threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()