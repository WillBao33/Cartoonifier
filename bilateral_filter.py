import numpy as np
import cv2
import matplotlib.pyplot as plt
import time

# bilateral filter
# radius:size of the filter 
# sigma_color: range 
# sigma_space: spatial
# C: channel number
def bilateral_filter(image, radius, sigma_color, sigma_space):
    H, W = image.shape[0], image.shape[1]
    C = 1 if len(image.shape) == 2 else image.shape[2]
    image = image.reshape(H, W, C)
    output_image = image.copy()

    for i in range(radius, H - radius):
        for j in range(radius, W - radius):
            for k in range(C):
                weight_sum = 0.0
                pixel_sum = 0.0
                for x in range(-radius, radius + 1):
                    for y in range(-radius, radius + 1):
                        # spatial gaussian
                        spatial_weight = -(x ** 2 + y ** 2) / (2 * (sigma_space ** 2))
                        # brightness gaussian
                        color_weight = -(int(image[i][j][k]) - int(image[i + x][j + y][k])) ** 2 / (2 * (sigma_color ** 2))
                        # 
                        weight = np.exp(spatial_weight + color_weight)
                        # total weight before normalization
                        weight_sum += weight
                        pixel_sum += (weight * image[i + x][j + y][k])
                # normalization
                value = pixel_sum / weight_sum
                output_image[i][j][k] = value
                print(output_image[i][j][k], '->', value)
    return output_image.astype(np.uint8)


if __name__ == '__main__':
    file_path = '/home/will/cv_course/test_1.jpg'
    image = cv2.imread(file_path, 1)
    # 高斯滤波
    scale_percent = 20 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
 
# resize image
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    #gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gauss = cv2.GaussianBlur(resized, (5, 5), 1)
    # 双边滤波
    start = time.time()
    bilateral = cv2.bilateralFilter(resized, d=5, sigmaColor=15, sigmaSpace=10)
    mat = bilateral_filter(resized, radius=5, sigma_color=15, sigma_space=10)
    end = time.time()
    print('processing time = {} min {} s'.format(int((end - start) / 60), int((end - start) % 60)))

    plt.figure(figsize=(10, 10))
    plt.suptitle('bilateral filter')
    plt.subplot(2, 2, 1), plt.title('image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.axis('off')
    plt.subplot(2, 2, 2), plt.title('my gaussian filter')
    plt.imshow(cv2.cvtColor(gauss, cv2.COLOR_BGR2RGB)), plt.axis('off')
    plt.subplot(2, 2, 3), plt.title('bilateral filter in opencv')
    plt.imshow(cv2.cvtColor(bilateral, cv2.COLOR_BGR2RGB)), plt.axis('off')
    plt.subplot(2, 2, 4), plt.title('my bilateral filter')
    plt.imshow(cv2.cvtColor(mat, cv2.COLOR_BGR2RGB)), plt.axis('off')
    plt.show()