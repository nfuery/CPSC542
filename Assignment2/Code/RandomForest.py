# Import libraries
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import matplotlib.pyplot as plt

# Load data
print("Loading images...")
IMAGE_PATH = "/app/rundir/images"
MASK_PATH = "/app/rundir/masks"

image_files = sorted(os.listdir(IMAGE_PATH))
mask_files = sorted(os.listdir(MASK_PATH))

# Preprocessing
print("Preprocessing images and masks...")
images = []
masks = []

for i, (img_file, mask_file) in enumerate(zip(image_files, mask_files)):
    print(f"Processing image {i+1}/{len(image_files)}")
    img_path = os.path.join(IMAGE_PATH, img_file)
    mask_path = os.path.join(MASK_PATH, mask_file)
    
    # Read and resize image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (512, 512))  # Resize to a consistent size
    images.append(img.flatten())  # Flatten image into 1D array
    
    # Read and resize mask
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (512, 512))  # Resize to match image size
    masks.append(mask.flatten())  # Flatten mask into 1D array

X = np.array(images)
y = np.array(masks)

# Train-Validation-Test Split
print("Splitting data into train, validation, and test sets...")
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

# # Model Training
# print("Training random forest classifier...")
# rf_classifier = RandomForestClassifier(n_estimators=1)
# rf_classifier.fit(X_train, y_train)

# # Save the trained model
# model_file_path = "/app/rundir/RF.joblib"
# dump(rf_classifier, model_file_path)
# print("Trained model saved successfully to:", model_file_path)

# Load the trained model
model_file_path = "/app/rundir/model.joblib"
rf_classifier = load(model_file_path)
print("Trained model loaded successfully from:", model_file_path)

# Model Evaluation
print("Evaluating model on validation set...")
y_val_pred = rf_classifier.predict(X_val)

# Calculate accuracy
y_val_flat = y_val.reshape(-1)
y_val_pred_flat = y_val_pred.reshape(-1)
accuracy = accuracy_score(y_val_flat, y_val_pred_flat)
report = classification_report(y_val_flat, y_val_pred_flat)

print("Validation Accuracy:", accuracy)
print("Classification Report:\n", report)

# Prediction and Visualization (if needed)
# You can add visualization code here if desired

# Optionally, you can load the trained model back later

def save_predictions(images, masks, predicted_masks, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    n_samples = min(len(images), len(masks), len(predicted_masks))
    
    for i in range(n_samples):
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        
        axes[0].imshow(images[i].reshape(512, 512, 3))
        axes[0].set_title("Image")
        axes[0].axis('off')
        
        axes[1].imshow(masks[i].reshape(512, 512), cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')
        
        axes[2].imshow(predicted_masks[i].reshape(512, 512), cmap='gray')
        axes[2].set_title("Predicted Mask")
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"visualization_{i}.png"))
        plt.close()

# Predict on validation set and save visualizations
print("Predicting on validation set...")
y_val_pred = rf_classifier.predict(X_val)
save_predictions(X_val, y_val, y_val_pred, "validation_visualizations")

# Predict on test set and save visualizations
print("Predicting on test set...")
y_test_pred = rf_classifier.predict(X_test)
save_predictions(X_test, y_test, y_test_pred, "test_visualizations")