import cv2
import numpy as np
from matplotlib import pyplot as plt

# task 10. Используя функцию fft2 постройте Фурье-спектр для заданного изображения.
original = cv2.imread("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_9\\images\\3.jpg")
# plt.imshow(original)
# plt.show()

ar = np.fft.fft2(original) # прямое двумерное дискретное преобразование Фурье
result = np.absolute(ar) # избавляемся от мнимой части
# plt.imshow(result)
# plt.show()

# task 9. Используя стандартную функцию быстрого преобразования Фурье, постройте графики вещественной и мнимой части Фурье-образов для функций
a = 9
b = 15

x = np.random.random_sample(256)
first = np.exp((-a**2)*(x**2)) # первая функция
result_1 = np.fft.fft(first) # прямое одномерное дискретное преобразование Фурье

second = (1/(1+(b**2)*(x**2))) # вторая функция
result_2 = np.fft.fft(second) # прямое одномерное дискретное преобразование Фурье

third = ((np.sin(a*x)/(1+(b*(x**2))))) # третья функция
result_3 = np.fft.fft(third) # прямое одномерное дискретное преобразование Фурье


plt.plot(result_1, color = 'b')
plt.title("График первой функции")
plt.show()
plt.plot(result_2, color = 'g')
plt.title("График второй функции")
plt.show()
plt.plot(result_3, color = 'r')
plt.title("График третьей функции")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
