import random
import numpy as np

# noises

def salt_pepper_noise(img):
    row, col = img.shape
    selected_pixel=random.randint(100,5000)
    for i in range(selected_pixel):
        # set these pixel to white
        x=random.randint(0,col-1)
        y=random.randint(0,row-1)
        img[y][x]=255
    for i in range(selected_pixel):
        # set these pixel to black
        x=random.randint(0,col-1)
        y=random.randint(0,row-1)
        img[y][x]=0
    return img

def gussian_noise(img, row, col):

    mean = 0.0
    std = 15.0
    noise = np.random.normal(mean, std, size=(row, col))
    img_noisy = np.add(img,noise)
    img_noisy = img_noisy.astype(np.uint8)
    return img_noisy


def uniform_noise(img, row, col):

    noise=np.random.uniform(-10, 10, size=(row, col))
    img_noisy = np.add(img, noise)
    img_noisy = img_noisy.astype(np.uint8)
    return img_noisy

#convolution
def apply_mask(img, mask, row, col):

    img_masked= np.zeros([row, col])
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            temp = img[i - 1, j - 1] * mask[0, 0] + img[i - 1, j] * mask[0, 1] + img[i - 1, j + 1] * mask[0, 2]+ img[
                i, j - 1] * mask[1, 0] + img[i, j] * mask[1, 1] + img[i, j + 1] * mask[1, 2] + img[i + 1, j - 1] * mask[
                       2, 0] + img[i + 1, j] * mask[2, 1] + img[i + 1, j + 1] * mask[2, 2]

            img_masked[i, j] = temp

    img_masked = img_masked.astype(np.uint8)
    return img_masked

# filters
# all filters are of size 3x3
def ave_filter(img, row, col):
    mask = np.ones([3, 3], dtype = int)
    mask = mask/3
    return apply_mask(img, mask, row, col)


def gaussian_filter(img, row, col):
    mask = np.array([[1, 2, 1],
                     [2, 4, 2],
                     [1, 2, 1]])
    mask = mask / 16
    return apply_mask(img, mask, row, col)


def median_filter(img, row, col):
    img_masked = np.zeros([row, col])
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            median_array = [img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1],
                            img[i, j - 1], img[i, j], img[i, j + 1],
                            img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1]]

            img_masked[i, j] = np.median(median_array)

    img_masked = img_masked.astype(np.uint8)
    return img_masked


# not working
def sobel_edge(img, row, col):
    mask_x = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
    mask_y = (mask_x.transpose())
    blurred_img = gaussian_filter(img, row, col)
    mask_x_dirc=apply_mask(blurred_img, mask_x, row, col)
    mask_y_dirc=apply_mask(blurred_img, mask_y, row, col)
    gradient =np.sqrt(np.square(mask_x_dirc) + np.square(mask_y_dirc))
    masked_img = (gradient * 255.0) / gradient.max()

    return np.uint8(masked_img)

def roberts_edge(img, row, col):
    mask_x = np.array([[0, 0, 0],
                       [0, 1, 0],
                       [0, 0, -1]])

    mask_y = np.array( [[0, 0, 0],
                        [0, 0, 1],
                        [0, -1, 0]])

    blurred_img = gaussian_filter(img, row, col)
    mask_x_dirc = apply_mask(blurred_img, mask_x, row, col)
    mask_y_dirc = apply_mask(blurred_img, mask_y, row, col)
    gradient =np.sqrt(np.square(mask_x_dirc) + np.square(mask_y_dirc))
    masked_img = (gradient * 255.0) / gradient.max()

    return np.uint8(masked_img)



# not working
def perwitt_edge(img, row, col):
    mask_x = np.array([[-1, -1, -1],
                       [0, 0, 0],
                       [1, 1, 1]])
    mask_y = (mask_x.transpose())

    blurred_img = gaussian_filter(img, row, col)
    mask_x_dirc = apply_mask(blurred_img, mask_x, row, col)
    mask_y_dirc = apply_mask(blurred_img, mask_y, row, col)
    gradient =np.sqrt(np.square(mask_x_dirc) + np.square(mask_y_dirc))
    masked_img = (gradient * 255.0) / gradient.max()

    return np.uint8(masked_img)


def canny_edge(img, row, col):
    pass
# multi stage filter
# 1) gaussian filter
# 2) sobel filter (mag ,dirc)
# 3) Non-max Suppression
# 4) Apply thresholding/hysteresis


def element_freq(arr):
    elements_count = {}
    if np.ndim(arr) > 1:
        arr = arr.flatten()
    for element in arr:   # iterating over the elements for frequency
        if element in elements_count: # checking whether it is in the dict or not
            elements_count[element] += 1 # incerementing the count by 1
        else:
            elements_count[element] = 1  # setting the count to 1

    return elements_count

def histogram(img):

    element_cumulative_sum = []
    cumulative_temp = 0
    scaled_cumulative_sum = []
    int_scaled_cumulative_sum = []

    # 1)count each pixel
    elements_count = element_freq(img)
    # 2)cumulative summation
    for key in elements_count:
        cumulative_temp = cumulative_temp + elements_count[key]
        element_cumulative_sum.append(cumulative_temp)

    # sanity check
    if cumulative_temp == element_cumulative_sum[-1]:
        print(True)

    # 3)scale
    scale_factor = (255) / (element_cumulative_sum[-1]-element_cumulative_sum[0])

    for i in range(0, len(element_cumulative_sum)):
        scale_element = (element_cumulative_sum[i]-element_cumulative_sum[0]) * scale_factor
        scaled_cumulative_sum.append(scale_element)

    # 4)round the numbers
    for i in range(0, len(scaled_cumulative_sum)):
        int_scaled_cumulative_sum.append(round(scaled_cumulative_sum[i]))
    # sanity check
    if max(int_scaled_cumulative_sum) == 255:
        print(True)

    # 5)draw between the numbers
    histogram_elements = element_freq(int_scaled_cumulative_sum)

    return histogram_elements.keys(),  histogram_elements.values()
    # keys, values = func.histogram(image)
    # plt.bar(keys, values)
    # plt.xticks(np.arange(0, max(keys), 25))
    # plt.yticks(np.arange(0, max(values) + 2, 1))
    # plt.show()

def equalized_img(img):
    pass


def normalize(img, display):
    Max = np.max(img)
    Min = np.min(img)
    normalized_img = np.array([(x - Min) / (Max - Min) for x in img])
    # sanity check
    if max(normalized_img.flatten()) <= 1 and min(normalized_img.flatten()) >= 0:
        print(True)

    if display == True:
        normalized_img *= 255
        normalized_img = normalized_img.astype(np.uint8)

    return normalized_img


def global_Thresholding(img, row, col):
    threshold_value = 127
    new_img = np.zeros([row, col])
    for i in range(1, row - 1):
        for j in range(1, col - 1):
            if img[i][j] >= threshold_value:
                new_img[i][j] = 255
            else:
                new_img[i][j] = 0

    new_img = new_img.astype(np.uint8)
    return new_img


def local_Thresholding(img, row, col):
    new_img = np.zeros([row, col])

    for i in range(1, row - 1):
        for j in range(1, col - 1):
            print('j= ',j)
            print('col= ',col-1)
            partial_img_array = [img[i - 1, j - 1], img[i - 1, j], img[i - 1, j + 1],
                                 img[i, j - 1], img[i, j], img[i, j + 1],
                                 img[i + 1, j - 1], img[i + 1, j], img[i + 1, j + 1]]

            mean_threshold = np.mean(partial_img_array)
            print(type(partial_img_array))
            print(partial_img_array)
            for element in (partial_img_array):
                if element >= int(mean_threshold):
                    new_img[i][j] = 255
                else:
                    new_img[i][j] = 0

    new_img = new_img.astype(np.uint8)

    # calculate the threshold which is the he weighted sum of neighbourhood values where weights are a gaussian window.
    # gaussian_window = np.array([[1, 2, 1],
    #                             [2, 4, 2],
    #                             [1, 2, 1]])
    # gaussian_window = gaussian_window / 16
    # gaussian_threshold = np.dot(gaussian_window.flatten(), partial_img_array.flatten())

    return new_img
