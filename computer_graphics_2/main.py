import cv2
import numpy as np

image = cv2.imread("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_2\\images\\9.jpg")
cv2.imshow("Image", image)
print("Исходный массив \n", image)

arr = np.diff(image)

print("Результат \n", arr)

maxi = np.max(arr)
mini = np.min(arr)

print('Максимум', maxi)
print('Минимум', mini)

cv2.waitKey(0)
cv2.destroyAllWindows()