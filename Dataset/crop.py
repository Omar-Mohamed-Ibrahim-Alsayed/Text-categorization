import os
from PIL import Image

# Define the source directory
source_dir = 'HW'

# Define the cropping box (left, upper, right, lower)
crop_box = (0, 600, 0, -800)  # Crop 600 pixels from the top and 800 pixels from the bottom

# Iterate over each file in the source directory
for file_name in os.listdir(source_dir):
    source_file = os.path.join(source_dir, file_name)

    # Check if the file is an image before processing
    if os.path.isfile(source_file) and file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        with Image.open(source_file) as img:
            # Get the size of the image
            width, height = img.size

            # Calculate the right and lower coordinates based on the image size
            right = width
            lower = height - 800

            # Define the cropping box for the current image
            crop_box = (0, 600, right, lower)

            # Crop the image
            cropped_img = img.crop(crop_box)
            
            # Save the cropped image, replacing the original
            cropped_img.save(source_file)

print("Images cropped successfully.")
