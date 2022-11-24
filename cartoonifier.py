import cv2
import matplotlib.pyplot as plt
import numpy as np
#from scipy import ndimage, signal
#import skimage
import argparse
import time
#from PIL import Image
from gaussian_kernel import gaussian_filter
from bilateral_filter import bilateral_filter
from non_max_suppression import non_max_suppression
from double_threshold import threshold
from edge_tracking import hysteresis



def get_args():
    parser = argparse.ArgumentParser(description='Image Cartoonifier')
    parser.add_argument('--image', '-i', default='image.png', metavar='FILE', help='Specify the image that will be cartoonized')
    parser.add_argument('--output', '-o', default='image.png', metavar='FILE', help='Specify the cartoonized image output name and location')
    return parser.parse_args()
#read image

if __name__ == '__main__':
    args = get_args()
    image = args.image
    out_image = args.output
    #print(image)
    image = cv2.imread(f'{image}',1)
    #cv2.imshow('original',image)

    #resize image
    scale_percent = 20 # percent of original size
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    gray_img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    #gaussian filter
    gaussian_kernel = gaussian_filter(11,1)
    gaussian_blur = cv2.filter2D(gray_img,-1,gaussian_kernel)
    gaussian_blur = np.float64(gaussian_blur)

    #sobel filter
    horizon_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]],np.float64)
    vertical_kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]],np.float64)
    horizon_conv = cv2.filter2D(gaussian_blur,-1,horizon_kernel)
    vertical_conv = cv2.filter2D(gaussian_blur,-1,vertical_kernel)
    gradient_mag = np.hypot(horizon_conv,vertical_conv)
    gradient_mag = gradient_mag/gradient_mag.max() * 255
    gradient_theta = np.rad2deg(np.arctan2(vertical_conv,horizon_conv))

    #nms
    gradient_nms = non_max_suppression(gradient_mag,gradient_theta)

    #double threshold
    non,weak,strong = threshold(gradient_nms)

    #edge tracking by hysteresis
    edge = hysteresis(non,weak,strong=255)
    edge = np.array(edge,dtype=np.uint8)
    (thresh, im_bw) = cv2.threshold(edge, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    edge = cv2.bitwise_not(im_bw) #reverse 0 and 255

    #bilateral filter on original image
    start = time.time()
    mat = bilateral_filter(resized,radius=5,sigma_color=15,sigma_space=10)
    end = time.time()
    print('processing time = {} min {} s'.format(int((end - start) / 60), int((end - start) % 60)))
    
    #bitwise_and operation
    cartoon=cv2.bitwise_and(mat,mat,mask=edge)
    cv2.imshow("cartoonized",cartoon)
    cv2.imwrite(f'{out_image}',cartoon)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

