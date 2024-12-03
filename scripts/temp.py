import sys
import os

# Add the project root directory to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib.pyplot as plt
from models import initialize_maf
from scripts import generate_synthetic_negatives

def visualize_synthetic_patches(patches, patch_size):
    """
    Visualize synthetic negative patches.

    Parameters:
    - patches (torch.Tensor): Tensor of synthetic patches with shape (N, C * H * W).
    - patch_size (int): Size of the patch (H and W are assumed equal).
    - num_to_display (int): Number of patches to display.
    """
     # Reshape patches to (N, C, H, W)
    patches = patches.view(-1, 3, patch_size, patch_size)
    
    # Convert to CPU if necessary
    patches = patches.cpu().detach()

    # Calculate grid size for visualization
    num_patches = len(patches)
    cols = 5  # Number of columns in the grid
    rows = (num_patches + cols - 1) // cols  # Calculate rows to fit all patches

    # Create the grid of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_patches:
            # Extract and normalize each patch
            patch = patches[i].permute(1, 2, 0).numpy()  # Convert to HWC format
            patch = (patch - patch.min()) / (patch.max() - patch.min())  # Normalize to [0, 1]
            ax.imshow(patch)
        else:
            # Hide empty subplots
            ax.axis("off")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# Example Usage
patch_size = 4  # Patch size (e.g., for a 32x32 image patch)
input_dim = patch_size*patch_size*3  # Input dimension (flattened patch size)
hidden_features = 256  # Number of hidden features in the MAF
num_transforms = 5  # Number of MAF layers (stacked transforms)
maf = initialize_maf(input_dim, hidden_features, num_transforms)

# Generate synthetic negatives
synthetic_negatives = generate_synthetic_negatives(maf, 20, [3,patch_size,patch_size], -50)
print(synthetic_negatives[0].shape)
visualize_synthetic_patches(synthetic_negatives, patch_size)
