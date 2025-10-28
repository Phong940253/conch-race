import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
try:
    haystack_img = cv2.imread('test2.png', cv2.IMREAD_UNCHANGED)
    needle_img = cv2.imread('increase.png', cv2.IMREAD_UNCHANGED)

    if haystack_img is None:
        print("Error: Could not read haystack image 'conch-race/test2.png'")
        exit()
    if needle_img is None:
        print("Error: Could not read needle image 'conch-race/increase.png'")
        exit()

except Exception as e:
    print(f"An error occurred while reading the images: {e}")
    exit()

# Handle alpha channel if present
if haystack_img.shape[2] == 4:
    haystack_img = cv2.cvtColor(haystack_img, cv2.COLOR_BGRA2BGR)
if needle_img.shape[2] == 4:
    needle_img = cv2.cvtColor(needle_img, cv2.COLOR_BGRA2BGR)

# Define the region of interest (ROI)
x, y, w, h = 506, 194, 252, 172
haystack_roi = haystack_img[y:y+h, x:x+w]

# Perform template matching
# Using TM_CCOEFF_NORMED for robust matching
result = cv2.matchTemplate(haystack_roi, needle_img, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

print(f"Max correlation value: {max_val}")
print(f"Match location in ROI: {max_loc}")

# Get the dimensions of the needle image
needle_w = needle_img.shape[1]
needle_h = needle_img.shape[0]

# Define the top-left and bottom-right corners of the rectangle
top_left = max_loc
bottom_right = (top_left[0] + needle_w, top_left[1] + needle_h)

# Draw a rectangle on the ROI for visualization
# Create a copy to avoid modifying the original ROI slice
haystack_roi_with_rect = haystack_roi.copy()
cv2.rectangle(haystack_roi_with_rect, top_left, bottom_right, (0, 255, 0), 2)

# Display the result using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(haystack_roi_with_rect, cv2.COLOR_BGR2RGB))
plt.title('Template Matching Result')
plt.axis('off')
plt.show()
