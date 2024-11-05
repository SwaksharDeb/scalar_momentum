import os
import numpy as np
from PIL import Image
import re

def natural_sort_key(s):
    """
    Return a key for natural sorting of strings containing numbers and text.
    This will help sort filenames like '621_cine.png' in a natural order.
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def get_frame_number(filename):
    """
    Extract the frame number from the filename.
    Assumes filename format contains a number before '_cine.png'
    """
    match = re.search(r'(\d+)_cine\.png', filename)
    if match:
        return match.group(1)
    return None

def process_images(base_folder, output_folder, sequence_length=20):
    """
    Process images from training and validation folders, concatenating consecutive images
    within sequences of specified length. Images are normalized to range [0, 1].
    
    Args:
        base_folder (str): Path to the Nifty folder
        output_folder (str): Path to the output voxel_data folder
        sequence_length (int): Length of each sequence (default: 20)
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Process both Training and Validation folders
    for folder_name in ['TrainingData', 'ValidationData']:
        folder_path = os.path.join(base_folder, folder_name)
        
        try:
            # Get sorted list of PNG files using natural sorting
            png_files = sorted(
                [f for f in os.listdir(folder_path) if f.endswith('.png')],
                key=natural_sort_key
            )
            
            print(f"Processing files in {folder_name}:")
            print("Sorted files:", png_files[:5], "..." if len(png_files) > 5 else "")
            
            # Process files in sequences
            for start_idx in range(0, len(png_files), sequence_length):
                # Get the current sequence
                sequence_files = png_files[start_idx:start_idx + sequence_length]
                
                # Process each pair in the sequence
                for i in range(len(sequence_files)):
                    try:
                        # Read current image
                        img1_path = os.path.join(folder_path, sequence_files[i])
                        img1 = np.array(Image.open(img1_path).convert('L'))  # Convert to grayscale
                        
                        # For the last frame in sequence, pair with itself
                        if i == len(sequence_files) - 1:
                            img2 = img1  # Use the same image
                            next_file = sequence_files[i]  # Use same filename for naming
                        else:
                            # Read next image
                            img2_path = os.path.join(folder_path, sequence_files[i + 1])
                            img2 = np.array(Image.open(img2_path).convert('L'))
                            next_file = sequence_files[i + 1]
                        
                        # Normalize images to [0, 1] range
                        img1_normalized = img1.astype(np.float32) / 255.0
                        img2_normalized = img2.astype(np.float32) / 255.0
                        
                        # Stack the normalized images
                        combined_img = np.stack([img1_normalized, img2_normalized], axis=-1)
                        
                        # Get frame numbers for filename
                        frame1_num = get_frame_number(sequence_files[i])
                        frame2_num = get_frame_number(next_file)
                        
                        # Create output filename with new format
                        prefix = 'train' if folder_name == 'TrainingData' else 'val'
                        output_path = os.path.join(
                            output_folder, 
                            f'{prefix}_{frame1_num}_cine_{frame2_num}.npy'
                        )
                        
                        np.save(output_path, combined_img)
                        
                        if i % 10 == 0:  # Print progress every 10 files
                            print(f"Processed pair: {frame1_num} and {frame2_num}")
                        
                    except Exception as e:
                        print(f"Error processing files {sequence_files[i]} and {next_file}: {str(e)}")
                        continue
                    
        except Exception as e:
            print(f"Error processing folder {folder_name}: {str(e)}")
            continue

# Example usage
base_folder = 'Nifty'
output_folder = 'voxel_data'
process_images(base_folder, output_folder, sequence_length=20)