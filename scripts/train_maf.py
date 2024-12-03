import torch
import torch.optim as optim
from nflows.flows import Flow

def train_maf(maf, patch, device, lr=1e-3):
    """
    Trains the MAF on a single patch.

    Parameters:
    - maf (Flow): The MAF model.
    - patch (torch.Tensor): The patch used for training (1D flattened tensor).
    - device (torch.device): The device to use (CPU/GPU).
    - lr (float): Learning rate for the optimizer.

    Returns:
    - loss (float): Training loss for the current patch.
    """
    # Move MAF and patch to the appropriate device
    maf.to(device)
    patch = patch.to(device)

    # Define optimizer
    optimizer = optim.Adam(maf.parameters(), lr=lr)

    # Train on the single patch
    optimizer.zero_grad()
    loss = -maf.log_prob(patch.unsqueeze(0)).mean()  # Negative log-likelihood
    loss.backward()
    optimizer.step()

    return loss.item()
