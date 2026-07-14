# Image Filter CLI

A command-line utility for applying convolution matrices and standard image processing filters. This project relies on vectorized OpenCV operations (`cv2.filter2D`) to apply kernels efficiently, bypassing manual loop iterations for faster execution.

## Project Structure

- `main.py`: The command-line interface entry point. Handles argument parsing and filter mapping.
- `filters.py`: Contains the core mathematical operations and custom kernel definitions.

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- NumPy

Install the required libraries:

```pip install opencv-python numpy```

## Usage

Execute the program directly from the terminal by passing the input image path and the target filter name.

Syntax:
```python main.py <path_to_image> <filter_name>```

Examples:
```python main.py flowers.jpg sobel```
```python main.py flowers.jpg blur```
```python main.py flowers.jpg vertical_edge```

## Available Filters

- `sobel`: Computes edge gradient magnitude using Sobel X and Y operators.
- `blur`: Applies a standard 5x5 Gaussian blur to smooth the image.
- `monochrome`: Converts a colored image to grayscale.
- `vertical_edge`: Highlights vertical edges using a `[-1, 1]` kernel.
- `horizontal_edge`: Highlights horizontal edges using a corresponding column kernel.
- `negative`: Inverts the color channels of the image.
