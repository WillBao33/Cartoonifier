import cv2 
import numpy as np 

def edge_mask(img, line_size, blur_value):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  #gray_blur = cv2.medianBlur(gray, blur_value)
  gray_blur = cv2.GaussianBlur(gray,(5,5),0)
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges