import os
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
import random

# Directory to save generated images
output_dir = 'printed_examples'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List of font paths
font_dir = 'fonts'  # Directory where your font files are stored
font_files = [os.path.join(font_dir, font) for font in os.listdir(font_dir) if font.endswith('.ttf')]

# Image properties
image_size = (800, 200)  # Width, height of the generated images
background_color = (255, 255, 255)  # White background
text_color = (0, 0, 0)  # Black text

def extract_text_from_pdf(pdf_path, max_samples=200):
    # Open the PDF file
    pdf_document = fitz.open(pdf_path)
    text_samples = []

    # Iterate over each page in the PDF
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text = page.get_text()
        # Split text into lines and collect lines
        lines = text.split('\n')
        text_samples.extend(lines)

    pdf_document.close()

    # Randomly sample from the extracted text
    return random.sample(text_samples, min(max_samples, len(text_samples)))

# Example usage
pdf_path = 'example.pdf'  # Path to your PDF file
text_samples = extract_text_from_pdf(pdf_path, max_samples=200)

# Generate images
for i, text in enumerate(text_samples):
    for font_file in font_files:
        try:
            # Load the font
            font = ImageFont.truetype(font_file, size=40)
            
            # Create a new image with white background
            img = Image.new('RGB', image_size, background_color)
            draw = ImageDraw.Draw(img)
            
            # Calculate text size and position using textbbox
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            text_x = (image_size[0] - text_width) // 2
            text_y = (image_size[1] - text_height) // 2
            
            # Draw the text on the image
            draw.text((text_x, text_y), text, font=font, fill=text_color)
            
            # Save the image
            font_name = os.path.basename(font_file).split('.')[0]
            output_path = os.path.join(output_dir, f'printed_{i}_{font_name}.png')
            img.save(output_path)
        except Exception as e:
            print(f"Error with font {font_file}: {e}")

print("Printed text images generated successfully.")
