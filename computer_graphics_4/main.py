import cv2
import math
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def adjust_gamma(image, gamma = 2.2):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def log_conversion(image):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    table = np.array([math.log10((i / 255.0) + 1) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def pixelVal(pix, r1, s1, r2, s2):
    if (0 <= pix and pix <= r1):
        return (s1 / r1) * pix
    elif (r1 < pix and pix <= r2):
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2


original = cv2.imread('C:\\Users\\bigbo\PycharmProjects\\computer_graphics_3\\images\\3.jpg')

_, threshold_image = cv2.threshold(original, 127, 255, 0)

negative = 255 - original

gam = adjust_gamma(original)

log = log_conversion(original)

r1 = 70
s1 = 0
r2 = 140
s2 = 255

# Vectorize the function to apply it to each value in the Numpy array.
pixelVal_vec = np.vectorize(pixelVal)
# Apply contrast stretching.
contrast_stretched = pixelVal_vec(original, r1, s1, r2, s2)

#cv2.imshow('Output', np.hstack((original, gam, log)))

titles = ['Original Image', 'Threshold Image',  'Negative Image', 'Gamma Correction', 'Logarithmic Conversion', 'Contrast Stretched']
images = [original, threshold_image, negative, gam, log, contrast_stretched]
for i in range(6):
    plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()