import cv2
import numpy as np
import sys
import os

def preprocess_image(image_path, output_path=None):
    """
    Preprocess an image: grayscale, resize, denoise, normalize, and detect edges.
    :param image_path: Path to the input image file.
    :param output_path: Optional path to save the processed image.
    :return: Processed image (numpy array).
    """
    # Validate file path
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Failed to read the image. Check file format and path.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize to fixed size (e.g., 256x256)
    resized = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_AREA)

    # Denoise using Gaussian blur
    denoised = cv2.GaussianBlur(resized, (1, 1), 0)

    # Normalize pixel values to range [0, 1]
    normalized = denoised.astype(np.float32) / 100.0

    # Edge detection (Canny)
    edges = cv2.Canny((normalized * 255).astype(np.uint8), 500, 200)

    # Save processed image if output path is provided
    if output_path:
        cv2.imwrite(output_path, edges)

    return edges

if __name__ == "__main__":

    try:
        processed_img = preprocess_image('grid_removed.png', 'output2.png')

        # Display the processed image
        cv2.imshow("Processed Image", processed_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
