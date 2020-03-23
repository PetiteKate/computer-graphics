import cv2
import numpy as np
import scipy.stats as ss
from matplotlib import pyplot as plt

original = cv2.imread('C:\\Users\\bigbo\PycharmProjects\\computer_graphics_5\\images\\4.jpg')

# lab 5
hist=plt.hist(original.ravel(),256,[0,256])
plt.show()

img_yuv = cv2.cvtColor(original, cv2.COLOR_BGR2YUV)
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

#vis = np.concatenate((original, img_output), axis=1)
#cv2.imshow("All", vis)

# lab 6
hist, bins = np.histogram (original.flatten(), 256, [0,256])
cdf = hist.cumsum()
cdf_normalized = cdf * float (hist.max()) / cdf.max()
plt.plot (cdf_normalized, color = 'b' )
plt.hist (original.flatten (), 256, [0,256], color = 'r')
plt.show ()

hist, bins = np.histogram (original.flatten(), 256, [0,256])

P = ss.expon.fit(hist)
rX = np.linspace(0,10, 100)
rP = ss.expon.pdf(rX, *P)
plt.plot (rP, color = 'b' )
plt.hist (original.flatten (), 256, [0,256], color = 'g')
plt.show ()

cv2.waitKey(0)
cv2.destroyAllWindows()