import cv2

image = cv2.imread("./test_1.jpg",1)
cv2.imshow("img",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
