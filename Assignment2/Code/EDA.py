# Import libraries
import os
import cv2
import matplotlib.pyplot as plt
from joblib import load

# Function for visualizing and saving multiple images and masks
def visualize_multiple_images(image_paths, mask_paths, save_path):
    # Create subplots for images and masks
    fig, axes = plt.subplots(2, 6, figsize=(20, 5))
    
    # Iterate over the first 10 images and masks
    for i in range(6):
        # Load the image
        img = cv2.imread(image_paths[i])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load the mask
        mask = cv2.imread(mask_paths[i])
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
        # Plot the image
        axes[0, i].imshow(img_rgb)
        axes[0, i].set_title(f"Image {i}")
        axes[0, i].axis("off")
        
        # Plot the mask
        axes[1, i].imshow(mask_rgb)
        axes[1, i].set_title(f"Mask {i}")
        axes[1, i].axis("off")
    
    # Adjust layout and save the visualization as a PNG file
    plt.tight_layout()
    plt.savefig(save_path)
    
    # Show the visualization
    plt.show()

# Get paths to the first 10 images and masks
image_paths = [f"/app/rundir/images/{i}.png" for i in range(6)]
mask_paths = [f"/app/rundir/masks/{i}.png" for i in range(6)]

# Define the save path for the combined visualization
combined_save_path = "/app/rundir/combined_visualization.png"

# Visualize and save the first 10 images and masks into a single PNG file
visualize_multiple_images(image_paths, mask_paths, combined_save_path)

