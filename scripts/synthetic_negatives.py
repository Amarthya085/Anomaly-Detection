import torch

def paste_patch_on_image(image, synthetic_patch, x, y, patch_dim):
    """
    Pasts the synthetic patch onto the original image at the specified location.

    Parameters:
    - image (torch.Tensor): Original image tensor of shape (C, H, W).
    - synthetic_patch (torch.Tensor): Generated synthetic patch of shape (C, patch_h, patch_w).
    - x (int): Top-left x-coordinate of the patch location.
    - y (int): Top-left y-coordinate of the patch location.
    - patch_dim (tuple): Dimensions of the patch (channels, height, width).

    Returns:
    - torch.Tensor: Image with the synthetic patch pasted at the same location.
    """
    # Clone the original image to avoid modifying the input
    mixed_image = image.clone()

    # Extract patch dimensions
    channels, patch_h, patch_w = patch_dim

    assert synthetic_patch.shape == (channels, patch_h, patch_w), \
        f"Patch dimensions {synthetic_patch.shape} do not match expected {patch_dim}"

    mixed_image[:, x:x+patch_h, y:y+patch_w] = synthetic_patch
    return mixed_image
