import cv2
import numpy as np
from matplotlib.figure import Figure
from scipy import ndimage
import matplotlib.pyplot as plt

# lab 11
original = cv2.imread("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_11\\images\\1.jpg", 0)
cv2.imshow("Image", original)

gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Gray", gray_image)

gauss = cv2.GaussianBlur(original, (5,5),0) # Gaussian Filter
cv2.imshow("Gauss", gauss)
batterwort = cv2.bilateralFilter(original,9,75,75)
cv2.imshow("Batterwort", batterwort)
ideal = cv2.blur(original, (5,5))
cv2.imshow("Ideal", ideal)

# lab 12
lowpass = ndimage.gaussian_filter(original, 5)
gauss_highpass = original - lowpass
cv2.imshow("", gauss_highpass)

min = np.minimum(original.shape[0], original.shape[1])
img = cv2.resize(original, (min, min))
img[img >= 225] = 0
M, N = img.shape
# computing the 2-d fourier transformation of the image
fourier_image = np.fft.fft2(img)

# ideal low pass filter
u = np.array(range(0, M))
v = np.array(range(0, N))
idx = np.where(u > (M / 2))
u[idx] = u[idx] - M
idy = np.where(v > N / 2)
v[idy] = v[idy] - N
[V, U] = np.meshgrid(v, u)
D = (U ** 2 + V ** 2) ** (1 / 2)
# cutoff = 40
cutoff = [50, 40, 20, 10]

H1 = (D > 20)
G1 = H1 * fourier_image
imback1 = np.fft.ifft2(G1)
imback1 = np.uint8(np.real(imback1))
imback1[imback1 >= 225] = 0
cv2.imshow("Ideal highpass filter", imback1)

order = [1,2]
H3 = 1 / (1 +(cutoff[1]/D)**(2*order[0]))
G3 = H3 * fourier_image
imback2 = np.fft.ifft2(G3)
imback2 = np.uint8(np.real(imback2))
imback2[imback2 >= 225] = 0
cv2.imshow("Butterworth highpass Filter", imback2)


cv2.waitKey(0)
cv2.destroyAllWindows()
