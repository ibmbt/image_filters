import cv2
import numpy as np

# Load the image in grayscale
image = cv2.imread('flowers.jpg', cv2.IMREAD_GRAYSCALE)

# Define Sobel kernels for x and y directions
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Apply the Sobel filter in the x direction
gradient_x = cv2.filter2D(image, -1, sobel_x)

# Apply the Sobel filter in the y direction
gradient_y = cv2.filter2D(image, -1, sobel_y)

# Compute the magnitude of the gradient
magnitude = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
magnitude = np.uint8(magnitude)  # Convert to 8-bit image

# Display the results
cv2.imshow("Original Image", image)
cv2.imshow("Sobel X", gradient_x)
cv2.imshow("Sobel Y", gradient_y)
cv2.imshow("Edge Magnitude", magnitude)

cv2.waitKey(0)
cv2.destroyAllWindows()