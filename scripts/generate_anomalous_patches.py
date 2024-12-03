import torch
import gc

def generate_synthetic_negatives(maf, num_patches, patch_dim, likelihood_threshold=-50):
    """
    Generate synthetic negative patches from a MAF by sampling low-likelihood regions.

    Parameters:
    - maf (Flow): Pretrained or partially trained MAF model.
    - num_patches (int): Number of synthetic patches to generate.
    - input_dim (int): Dimensionality of each patch (e.g., flattened patch size).
    - likelihood_threshold (float): Log-likelihood threshold to reject high-likelihood samples.

    Returns:
    - torch.Tensor: A tensor of synthetic negative patches.
    """
    maf.to(device)
    synthetic_patches = []
    context = None  # Replace with actual context if needed
    samples = maf._sample(num_patches, context).to(device)

    generated_patches = samples.view(num_patches, *patch_dim)
    print("generated_patches shape:",generated_patches.shape)

    generated_patches_flat = generated_patches.view(num_patches, -1)
        
    # Compute log-likelihood for generated patches
    log_probs = maf._log_prob(generated_patches_flat, context)
    
    # Filter out high-likelihood samples
    low_likelihood_indices = (log_probs < likelihood_threshold).nonzero(as_tuple=True)[0]
    low_likelihood_patches = generated_patches[low_likelihood_indices]
    
    if len(low_likelihood_patches) == 0:
        raise ValueError("No patches found with log probability less than the threshold.")

    # Find the patch with the highest likelihood among the low-likelihood patches
    highest_likelihood_index = log_probs[low_likelihood_indices].argmax()
    print("hightest_likelihood_index - ", log_probs[highest_likelihood_index])
    highest_likelihood_patch = low_likelihood_patches[highest_likelihood_index]

    torch.cuda.empty_cache()
    gc.collect()

    # Return exactly `num_patches` synthetic negatives
    return highest_likelihood_patch.view(*patch_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
# Generate synthetic negatives
synthetic_negatives = generate_synthetic_negatives(maf, num_patches, input_dim, likelihood_threshold)

# Print details
print(f"Generated {synthetic_negatives.shape[0]} synthetic negative patches of size {patch_size}x{patch_size}.")
'''