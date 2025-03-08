'''
import numpy as np
import cv2


image = cv2.imread('flowers.jpg', cv2.IMREAD_GRAYSCALE)
output_image = image.copy()

# kernal 1
kernel_1 = np.array([[-1, 1]])
rows, cols = image.shape

for i in range(1, rows):
    for j in range(1, cols - 1):
        sum_value = (
            image[i][j-1] * kernel_1[0][0] +
            image[i][j] * kernel_1[0][1]
        )
        
        output_image[i][j] = sum_value
        copy_image1 = output_image.copy()

cv2.imshow("horizontal", output_image)

# kernel 2
output_image = image.copy()
kernel_2 = np.array([[-1], [1]])

for i in range(1, rows - 1):
    for j in range(0, cols):
        sum_value = (
            image[i-1][j] * kernel_2[0][0] +
            image[i][j] * kernel_2[1][0]
        )
        
        output_image[i][j] = sum_value
        copy_image2 = output_image.copy()

cv2.imshow("vertical", output_image)


# kernel 3
kernel_3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        sum_value = (
            image[i-1][j-1] * kernel_3[0][0] +
            image[i-1][j] * kernel_3[0][1] +
            image[i-1][j+1] * kernel_3[0][2] +
            image[i][j-1] * kernel_3[1][0] +
            image[i][j] * kernel_3[1][1] +
            image[i][j+1] * kernel_3[1][2] +
            image[i+1][j-1] * kernel_3[2][0] +
            image[i+1][j] * kernel_3[2][1] +
            image[i+1][j+1] * kernel_3[2][2]
        )

        output_image[i][j] = sum_value
        copy_image3 = output_image.copy()

cv2.imshow("Sobel X", output_image)
cv2.destroyAllWindows()

# kernel 4
kernel_4 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        sum_value = (
            image[i-1][j-1] * kernel_4[0][0] +
            image[i-1][j] * kernel_4[0][1] +
            image[i-1][j+1] * kernel_4[0][2] +
            image[i][j-1] * kernel_4[1][0] +
            image[i][j] * kernel_4[1][1] +
            image[i][j+1] * kernel_4[1][2] +
            image[i+1][j-1] * kernel_4[2][0] +
            image[i+1][j] * kernel_4[2][1] +
            image[i+1][j+1] * kernel_4[2][2]
        )
        output_image[i][j] = sum_value


cv2.imshow("Sobel Y", output_image)
cv2.destroyAllWindows()

'''

import numpy as np
import cv2

image = cv2.imread('flowers.jpg', cv2.IMREAD_GRAYSCALE)

rows, cols = image.shape


kernel_3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
f_x_sobel = np.zeros((rows, cols))

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        sum_value = (
            image[i-1, j-1] * kernel_3[0, 0] +
            image[i-1, j] * kernel_3[0, 1] +
            image[i-1, j+1] * kernel_3[0, 2] +
            image[i, j-1] * kernel_3[1, 0] +
            image[i, j] * kernel_3[1, 1] +
            image[i, j+1] * kernel_3[1, 2] +
            image[i+1, j-1] * kernel_3[2, 0] +
            image[i+1, j] * kernel_3[2, 1] +
            image[i+1, j+1] * kernel_3[2, 2]
        )
        f_x_sobel[i, j] = sum_value


kernel_4 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

f_y_sobel = np.zeros((rows, cols))

for i in range(1, rows - 1):
    for j in range(1, cols - 1):
        sum_value = (
            image[i-1, j-1] * kernel_4[0, 0] +
            image[i-1, j] * kernel_4[0, 1] +
            image[i-1, j+1] * kernel_4[0, 2] +
            image[i, j-1] * kernel_4[1, 0] +
            image[i, j] * kernel_4[1, 1] +
            image[i, j+1] * kernel_4[1, 2] +
            image[i+1, j-1] * kernel_4[2, 0] +
            image[i+1, j] * kernel_4[2, 1] +
            image[i+1, j+1] * kernel_4[2, 2]
        )
        f_y_sobel[i, j] = sum_value


f_x_sobel = np.clip(f_x_sobel, -255, 255)
f_y_sobel = np.clip(f_y_sobel, -255, 255)


magnitude = np.sqrt(f_x_sobel**2 + f_y_sobel**2)
magnitude = np.clip(magnitude, 0, 255)
magnitude_display = np.uint8(magnitude)

cv2.imshow("Sobel X", f_x_sobel)
cv2.imshow("Sobel Y", f_y_sobel)
cv2.imshow("Magnitude", magnitude_display)
cv2.waitKey(0)
cv2.destroyAllWindows()