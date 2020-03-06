import cv2
import numpy as np
from matplotlib import pyplot as plt

original = cv2.imread('C:\\Users\\bigbo\PycharmProjects\\computer_graphics_5\\images\\4.jpg')

# lab 5
gist=plt.hist(original.ravel(),256,[0,256])
plt.show()

img_yuv = cv2.cvtColor(original, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

vis = np.concatenate((original, img_output), axis=1)
cv2.imshow("All", vis)

#lab 6
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 1000)
abs(mu - np.mean(s)) < 0.01
abs(sigma - np.std(s, ddof=1)) < 0.01
plt.plot(gist, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (gist - mu)**2 / (2 * sigma**2) ),linewidth=2, color='r')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()