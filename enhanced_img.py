import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import pickle

# Define path to your dataset
data_dir = 'Custom_Datatset'
image_paths = []

# List all subdirectories (categories)
categories = os.listdir(data_dir)

# Create a list to hold image paths
for label, category in enumerate(categories):
    category_path = os.path.join(data_dir, category)
    for image_file in os.listdir(category_path):
        image_path = os.path.join(category_path, image_file)
        image_paths.append(image_path)

# Define transformations for data preprocessing
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the saved ResNet18 model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 25)  # 25 classes
model.load_state_dict(torch.load('resnet18_finetuned.pth'))
model = model.to(device)
model.eval()

# Modify ResNet18 to extract features from the layer before the classification layer
feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove the last layer (fc layer)
feature_extractor = feature_extractor.to(device)

# Function to extract features using the modified ResNet18
def extract_features(feature_extractor, image_paths, transform):
    features = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            feature = feature_extractor(image)
            feature = feature.view(feature.size(0), -1)  # Flatten the feature map
            feature = nn.functional.adaptive_avg_pool1d(feature.unsqueeze(0), 256).squeeze(0)  # Adjust the feature dimension to 256
        features.append(feature.cpu().numpy().flatten())  # Flatten the features to 1D
    return features

# Extract features
resnet_features = extract_features(feature_extractor, image_paths, data_transforms)

# Generate random latent vectors (simulate StyleGAN3 latent vectors)
stylegan_latent_vectors = np.random.randn(len(image_paths), 256)

# Concatenate ResNet18 features and StyleGAN3 latent vectors
# Adjusting the size to 512 as expected by StyleGAN3
combined_features = [np.concatenate((resnet_feat, stylegan_latent), axis=0) for resnet_feat, stylegan_latent in zip(resnet_features, stylegan_latent_vectors)]

# Function to perform linear interpolation
def linear_interpolation(features, alpha=0.5):
    interpolated_features = []
    for i in range(len(features) - 1):
        interp_feat = alpha * features[i] + (1 - alpha) * features[i + 1]
        interpolated_features.append(interp_feat)
    return interpolated_features

# Perform linear interpolation on enhanced features
interpolated_features = linear_interpolation(combined_features)

# Load StyleGAN3 model
with open('~/training-runs/00064-stylegan3-r-Custom_dataset_sgan-gpus1-batch32-gamma2/network-snapshot-000004.pkl', 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

# Function to generate images using StyleGAN3
def generate_enhanced_images(features):
    enhanced_images = []
    for feature in features:
        stylegan_latent = torch.tensor(feature[:512], dtype=torch.float32).unsqueeze(0).cuda()  # Ensure the latent vector is in the correct format
        img = G(stylegan_latent, None)  # Generate image
        img = (img.clamp(-1, 1) + 1) / 2 * 255  # Convert to [0, 255] range
        img = img.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)  # Convert to HWC format and numpy array
        enhanced_images.append(Image.fromarray(img[0]))
    return enhanced_images

# Generate enhanced images
enhanced_images = generate_enhanced_images(interpolated_features)

# Save generated images to disk
output_dir = 'Enhanced_Images'
os.makedirs(output_dir, exist_ok=True)
for i, img in enumerate(enhanced_images):
    img.save(os.path.join(output_dir, f'enhanced_image_{i}.png'))

print("Enhanced images have been generated and saved.")
