import os
import shutil

# Define source and destination directories
source_base_dir = '.\data'
destination_dir = '.\HW'

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_dir):
    os.makedirs(destination_dir)

# Iterate over each directory from 001 to 671
for i in range(1, 672):
    # Construct the directory name
    dir_name = f'{i:03d}'
    source_dir = os.path.join(source_base_dir, dir_name)

    # Check if the source directory exists
    if os.path.exists(source_dir):
        # Iterate over each file in the directory
        for file_name in os.listdir(source_dir):
            source_file = os.path.join(source_dir, file_name)
            destination_file = os.path.join(destination_dir, file_name)

            # Check if it's a file before copying
            if os.path.isfile(source_file):
                shutil.copy(source_file, destination_file)

print("Files copied successfully.")
