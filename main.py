import argparse
import cv2
import sys
import filters

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Apply image processing filters.")
    parser.add_argument("image_path", help="Path to the input image.")
    parser.add_argument(
        "filter", 
        choices=[
            "sobel", "blur", "monochrome", 
            "vertical_edge", "horizontal_edge", "negative"
        ], 
        help="The specific filter to apply to the image."
    )

    args = parser.parse_args()

    # Load image (loads in color by default; filters handle grayscale conversion if needed)
    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not load image from {args.image_path}")
        sys.exit(1)

    # Map string arguments to function references
    filter_map = {
        "sobel": filters.sobel_filter,
        "blur": filters.blur_filter,
        "monochrome": filters.monochrome_filter,
        "vertical_edge": filters.vertical_edge_detector,
        "horizontal_edge": filters.horizontal_edge_detector,
        "negative": filters.negative_filter
    }

    # Execute the chosen filter
    selected_function = filter_map[args.filter]
    output_image = selected_function(image)

    # Display results
    cv2.imshow("Original", image)
    cv2.imshow(f"Filter applied: {args.filter}", output_image)
    
    print("Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
