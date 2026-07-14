import cv2
import numpy as np

def sobel_filter(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # cv2.CV_64F prevents overflow when calculating gradients
    gradient_x = cv2.filter2D(image, cv2.CV_64F, sobel_x)
    gradient_y = cv2.filter2D(image, cv2.CV_64F, sobel_y)

    magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    return np.uint8(np.clip(magnitude, 0, 255))

def blur_filter(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def monochrome_filter(image):
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def vertical_edge_detector(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    kernel = np.array([[-1, 1]])
    return cv2.filter2D(image, -1, kernel)

def horizontal_edge_detector(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    kernel = np.array([[-1], [1]])
    return cv2.filter2D(image, -1, kernel)

def negative_filter(image):
    return cv2.bitwise_not(image)