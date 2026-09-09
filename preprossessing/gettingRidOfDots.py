import cv2
import numpy as np

# Load image
image = cv2.imread("LR-pH4-CMC-6_23_26.png")

# Convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Detect blue/purple grid
lower = np.array([100, 40, 20])
upper = np.array([160, 255, 255])

grid_mask = cv2.inRange(hsv, lower, upper)

# Expand the mask slightly
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (3, 3)
)

grid_mask = cv2.dilate(grid_mask, kernel, iterations=1)

# Remove grid using inpainting
cleaned = cv2.inpaint(
    image,
    grid_mask,
    3,
    cv2.INPAINT_TELEA
)

# Save result
cv2.imwrite("grid_removed.png", cleaned)