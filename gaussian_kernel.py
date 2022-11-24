import numpy as np

def gaussian_filter(size,sigma):
    if size % 2 == 0:
        size += 1

    max_point = size // 2 
    min_point = -max_point 

    K = np.zeros((size, size))  # kernel matrix
    for x in range(min_point, max_point + 1):
        for y in range(min_point, max_point + 1):
            value = (1 / (2 * np.pi * (sigma ** 2)) * np.exp((-(x ** 2 + y ** 2)) / (2 * (sigma ** 2))))
            K[x - min_point, y - min_point] = value
            #print(K)

    return K