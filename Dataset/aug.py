import os
from PIL import Image, ImageEnhance, ImageOps
import random

# Directory containing images
input_dir = 'printed_examples'
output_dir = 'augmented_examples'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def apply_rotation(img, angle):
    return img.rotate(angle, expand=True)

def apply_rgb_modifications(img, r_factor=1.0, g_factor=1.0, b_factor=1.0):
    r, g, b = img.split()
    r = ImageEnhance.Brightness(r).enhance(r_factor)
    g = ImageEnhance.Brightness(g).enhance(g_factor)
    b = ImageEnhance.Brightness(b).enhance(b_factor)
    return Image.merge('RGB', (r, g, b))

def apply_pixelization(img, pixel_size=10):
    # Resize down
    small = img.resize((img.size[0] // pixel_size, img.size[1] // pixel_size), resample=Image.NEAREST)
    # Resize up
    return small.resize(img.size, Image.NEAREST)

def augment_image(image_path):
    with Image.open(image_path) as img:
        # Apply rotation
        #angle = random.randint(0, 360)
        #rotated_img = apply_rotation(img, angle)
        
        # Apply RGB color modifications
        r_factor = random.uniform(0.5, 1.5)
        g_factor = random.uniform(0.5, 1.5)
        b_factor = random.uniform(0.5, 1.5)
        rgb_modified_img = apply_rgb_modifications(img, r_factor, g_factor, b_factor)
        
        # Apply pixelization
        pixel_size = random.randint(1, 5)
        pixelized_img = apply_pixelization(rgb_modified_img, pixel_size)
        
        # Save the augmented image
        base_name = os.path.basename(image_path)
        output_path = os.path.join(output_dir, f'augmented_{base_name}')
        pixelized_img.save(output_path)

# Apply augmentations to all images in the input directory
for image_file in os.listdir(input_dir):
    image_path = os.path.join(input_dir, image_file)
    if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
        try:
            augment_image(image_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

print("Augmentation complete.")
