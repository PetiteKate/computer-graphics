import cv2
import numpy as np
import scipy.stats as ss
from matplotlib import pyplot as plt
from scipy import ndimage


original = cv2.imread("C:\\Users\\bigbo\PycharmProjects\\computer_graphics_13\\images\\2.jpg")
gray_image = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
cv2.imshow("Image", original)


lowpass = ndimage.gaussian_filter(original, 5)
gauss_highpass = original - lowpass
cv2.imshow("Highpass", gauss_highpass)


img_yuv = cv2.cvtColor(gauss_highpass, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

cv2.imshow("Result", img_output)

cv2.waitKey(0)
cv2.destroyAllWindows()