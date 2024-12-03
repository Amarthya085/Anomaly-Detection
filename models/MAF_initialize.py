import torch
from nflows.transforms import MaskedAffineAutoregressiveTransform
from nflows.distributions import StandardNormal
from nflows.flows import Flow

def initialize_maf(input_dim, hidden_features, num_transforms):
    """
    Initialized a Masked Autoregressive Flow (MAF) with random weights.

    Parameters:
    - input_dim (int): Dimensionality of the input (e.g., patch size squared for images).
    - hidden_features (int): Number of hidden features in each MAF layer.
    - num_transforms (int): Number of stacked MAF transforms.

    Returns:
    - Flow: Initialized MAF model.
    """
    # Create a list of Masked Affine Autoregressive Transforms
    transforms = []
    for _ in range(num_transforms):
        transforms.append(MaskedAffineAutoregressiveTransform(features=input_dim, hidden_features=hidden_features))
    
    # Combine transforms into a single sequential transform
    from nflows.transforms import CompositeTransform
    transform = CompositeTransform(transforms)

    # Define the base distribution (e.g., standard Gaussian)
    base_distribution = StandardNormal([input_dim])

    # Create the MAF model
    maf = Flow(transform, base_distribution)
    
    return maf

# Print MAF structure
#print(maf)
