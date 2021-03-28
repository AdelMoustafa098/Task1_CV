import func
import cv2
import numpy as np
import matplotlib.pylab as plt


image = cv2.imread('Lenna_(test_image).png',cv2.IMREAD_GRAYSCALE)
row, col = image.shape


# img_s_p_noise=func.salt_pepper_noise(image)
# image_uniform_noise=func.uniform_noise(image,row,col)
# image_gaussian_noise=func.gussian_noise(image, row, col)
# image_ave=func.ave_filter(image, row, col)
# image_gaussian=func.gaussian_filter(image,row,col)

# image_median=func.median_filter(img_s_p_noise, row, col)
# image_sobel =func.sobel_edge(image, row, col)
# image_perwitt =func.perwitt_edge(image, row, col)
image_robert=func.roberts_edge(image, row, col)
# norma_img = func.normalize(image)
# g_th = func.global_Thresholding(image, row, col)

# l_th = func.local_Thresholding(image, row, col)

# scale_percent =100# calculate the (scale_percent) percent of original dimensions
# width = int(image_sobel.shape[1] * scale_percent / 100)
# height = int(image_sobel.shape[0] * scale_percent / 100)
# dsize = (width, height)
# # image_sobel_scaled = cv2.resize(image_sobel, dsize)
# image_8 = cv2.resize(image, dsize)


# cv2.imshow('uniform', image_uniform_noise)
cv2.imshow('norm', image_robert)
cv2.imshow('Original', image)

cv2.waitKey(0)

# keys, values = func.histogram(image)
# plt.bar(keys, values)
# plt.xticks(np.arange(0, max(keys), 25))
# plt.yticks(np.arange(0, max(values) + 2, 1))
# plt.show()


