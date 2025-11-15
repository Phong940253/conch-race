import pyautogui
import matplotlib.pyplot as plt
import numpy as np
import cv2

def debug_image_location_from_file(haystack_path, needle_paths):
    """
    Loads a haystack image and highlights the locations of the needle images.
    """
    try:
        haystack_img = cv2.imread(haystack_path)
        if haystack_img is None:
            print(f"Error: Could not read haystack image '{haystack_path}'")
            return
    except Exception as e:
        print(f"An error occurred while reading the haystack image: {e}")
        return

    for needle_path in needle_paths:
        try:
            # Find all occurrences of the needle image within the haystack image
            locations = pyautogui.locateAll(needle_path, haystack_path, confidence=0.8)
            
            found = False
            for box in locations:
                # Draw a rectangle around each found image
                cv2.rectangle(haystack_img, (box.left, box.top), (box.left + box.width, box.top + box.height), (0, 255, 0), 2)
                found = True
            
            if found:
                print(f"Found '{needle_path}' in '{haystack_path}'.")
            else:
                print(f"Could not find '{needle_path}' in '{haystack_path}'.")

        except Exception as e:
            print(f"An error occurred while searching for '{needle_path}': {e}")

    # Display the result
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(haystack_img, cv2.COLOR_BGR2RGB))
    plt.title("Image Locations")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # The user should replace 'test.png' with the actual path to their haystack image.
    haystack_image_path = 'split-data/capture_20251115_110406.png'
    images_to_find = ['increase.png', 'confirm1.png', 'confirm2.png', 'refresh.png']
    debug_image_location_from_file(haystack_image_path, images_to_find)
