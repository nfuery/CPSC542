# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from torchvision import transforms
from pathlib import Path
from tqdm import tqdm
import os
from torch import optim

# Load image and mask paths
IMG_DIRECTORY = Path("/app/rundir/images")
MASK_DIRECTORY = Path("/app/rundir/masks")
IMG_FILE_PATHS = sorted(list(IMG_DIRECTORY.glob("*.png")))
MASK_FILE_PATHS = sorted(list(MASK_DIRECTORY.glob("*.png")))
print(f'Total Images = {len(IMG_FILE_PATHS)}')
print(f'Total Masks = {len(MASK_FILE_PATHS)}')

# Prepare data for training
image_file_paths = [img_path for img_path in IMG_FILE_PATHS]
mask_file_paths = [mask_path for mask_path in MASK_FILE_PATHS]
data_frame = pd.DataFrame({'Image': image_file_paths, 'Mask': mask_file_paths})

# Split data into train, validation, and test sets
train_data, remaining_data = train_test_split(data_frame, test_size=0.3)
validation_data, test_data = train_test_split(remaining_data, test_size=0.5)

# Function to map colors to IDs in masks
def color_to_id_mapping(image, colormap):
    image_array = np.array(image)
    mapped_matrix = np.full((image_array.shape[0], image_array.shape[1]), -1, dtype=np.int32)
    for h in range(image_array.shape[0]):
        for w in range(image_array.shape[1]):
            pixel = tuple(image_array[h, w])
            if pixel in colormap:
                mapped_matrix[h, w] = colormap[pixel]
    return mapped_matrix

# Dataset class for loading data
class ImageMaskDataset(Dataset):
    def __init__(self, data, colormap, image_transforms, mask_transforms):
        self.data = data
        self.colormap = colormap
        self.image_transforms = image_transforms
        self.mask_transforms = mask_transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx, 0]
        mask_path = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")
        image = self.image_transforms(image)
        mask = self.mask_transforms(mask)
        mask = color_to_id_mapping(mask, self.colormap)
        return image, mask

# Setup transformations, datasets, and dataloaders
image_augmentations = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()])
mask_augmentations = transforms.Compose([transforms.Resize((512, 512))])
training_dataset = ImageMaskDataset(train_data, color_to_id_mapping, image_augmentations, mask_augmentations)
validation_dataset = ImageMaskDataset(validation_data, color_to_id_mapping, image_augmentations, mask_augmentations)
training_loader = DataLoader(training_dataset, batch_size=1, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)

# Initialize model, loss function, and optimizer
device_type = "cpu"
network_model = smp.Unet(classes=5).to(device_type)
loss_function = smp.losses.DiceLoss(mode="multiclass", classes=5)
optimizer = optim.Adam(network_model.parameters(), lr=0.01, weight_decay=0.0001)

# Training and validation functions
def training_cycle(network_model, loader, loss_function, optimizer):
    network_model.train()
    total_loss = 0
    for images, masks in tqdm(loader, desc="Training"):
        images, masks = images.to(device_type), masks.to(device_type)
        optimizer.zero_grad()
        predictions = network_model(images)
        loss = loss_function(predictions, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validation_cycle(network_model, loader, loss_function):
    network_model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validation"):
            images, masks = images.to(device_type), masks.to(device_type)
            predictions = network_model(images)
            loss = loss_function(predictions, masks)
            total_loss += loss.item()
    return total_loss / len(loader)

# Run training and validation for each epoch
epochs = 10
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    train_loss = training_cycle(network_model, training_loader, loss_function, optimizer)
    validation_loss = validation_cycle(network_model, validation_loader, loss_function)
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Validation Loss: {validation_loss:.4f}")

# Prediction function
def generate_predictions(network_model, loader):
    network_model.eval()
    prediction_list = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Predicting"):
            images = images.to(device_type)
            outputs = network_model(images)
            _, predictions = torch.max(outputs, 1)
            prediction_list.append(predictions.cpu())
    return torch.cat(prediction_list)

test_data_loader = DataLoader(test_data, batch_size=1, shuffle=False)  
test_predictions = generate_predictions(network_model, test_data_loader)

test_dataset = ImageMaskDataset(test_data, color_to_id_mapping, image_augmentations, mask_augmentations) 

# Function for plotting predictions alongside their corresponding images
def display_prediction_samples(dataset, predictions, num_samples=5, save_directory="/app/rundir"):
    for i in range(num_samples):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        image, _ = dataset[i]
        image = image.cpu().numpy().transpose(1, 2, 0)
        prediction = predictions[i].cpu().numpy()
        
        axs[0].imshow(image)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        
        axs[1].imshow(prediction, cmap='gray')
        axs[1].set_title("Predicted Mask")
        axs[1].axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(save_directory, f"sample_prediction_{i+1}.png")
        plt.savefig(save_path)
        plt.close()

display_prediction_samples(test_dataset, test_predictions, num_samples=5, save_directory="/app/rundir")
