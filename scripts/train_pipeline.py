import torch
import torch.optim as optim
import random
from torchvision import datasets, transforms
from train_maf import train_maf 
from models import initialize_maf
from scripts import generate_synthetic_negatives
from synthetic_negatives import paste_patch_on_image  
import math

def load_data(dataset_path, batch_size, crop_size):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_maf_on_patch(maf, image, patch_dim, x, y, device):
    # Extract a random patch from the image
    patch = image[:, x:x + patch_dim[1], y:y + patch_dim[2]].flatten()  # Flatten to 1D vector

    # Training the MAF on the patch
    loss = train_maf(maf, patch, device)
    return loss

def generate_and_paste_patch(maf, image, patch_size, x, y, device):
    # Generating a synthetic patch from the MAF
    synthetic_patch = generate_synthetic_negatives(maf, 20, [3,patch_size,patch_size], -50)

    # Pasting the synthetic patch onto the image at the same location as the original patch
    #paste_patch_on_image(image, synthetic_patch, x, y, patch_dim)
    mixed_image = paste_patch_on_image(image, synthetic_patch, x, y, 3*patch_size*patch_size)
    return mixed_image

def train_pipeline(maf, segmenter, dataset_path, patch_dim, device, epochs=10, batch_size=8, warmup_epochs=2):
    dataloader = load_data(dataset_path, batch_size, patch_dim[1])

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")

        # Only train MAF during the first `warmup_epochs`
        if epoch < warmup_epochs:
            print("Training MAF...")
            for images, _ in dataloader:
                images = images.to(device)
                for img in images:
                    x = random.randint(0, img.shape[1] - patch_dim[1])
                    y = random.randint(0, img.shape[2] - patch_dim[2])
                    loss = train_maf_on_patch(maf, img, patch_dim, device)
                    print(f"MAF Loss: {loss:.4f}")
        else:
            print("Training with Mixed-Content Images...")
            for images, _ in dataloader:
                images = images.to(device)
                for img in images:
                    # Training MAF and sample a synthetic patch
                    mixed_image = generate_and_paste_patch(maf, img, math.sqrt(patch_dim/3), device)

                    # Fine-tune the segmentation model on the mixed-content image
                    # Implement fine-tuning logic (e.g., forward pass, loss calculation, backprop)
                    fine_tune_segmentation(segmenter, mixed_image, device)
