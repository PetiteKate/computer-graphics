import cv2
import numpy as np
import matplotlib.pyplot as plt


original = cv2.imread("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_11\\images\\6.jpg", 0)
cv2.imshow("original", original)
ret, imgf = cv2.threshold(original, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Otsu", imgf)

plt.subplot(3, 1, 1), plt.imshow(original, cmap='gray')
plt.title('Original Noisy Image'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 1, 2), plt.hist(original.ravel(), 256)
plt.axvline(x=ret, color='r', linestyle='dashed', linewidth=2)
plt.title('Histogram'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 1, 3), plt.imshow(imgf, cmap='gray')
plt.title('Otsu'), plt.xticks([]), plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()