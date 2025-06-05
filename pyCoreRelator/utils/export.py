"""
Export and file creation functions
"""

import os
from PIL import Image


def create_gif(frame_folder, output_filename, duration=300):
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.png')])
    
    if not frame_files:
        return f"No frames found for GIF creation in {frame_folder}"
    
    # Open first image to get dimensions and mode
    first_img = Image.open(os.path.join(frame_folder, frame_files[0]))
    
    # Create GIF with append_images to avoid loading all frames at once
    frames_iterator = (Image.open(os.path.join(frame_folder, f)) for f in tqdm(frame_files[1:], 
                        desc=f"Processing frames for {output_filename}"))
    
    first_img.save(
        output_filename,  # Make sure this is a string, not a tuple
        format='GIF',
        append_images=frames_iterator,
        save_all=True,
        duration=duration,
        loop=0,
        optimize=False  # Faster processing
    )
    
    return f"Created GIF with {len(frame_files)} frames at {output_filename}"