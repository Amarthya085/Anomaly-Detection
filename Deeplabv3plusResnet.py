

import torch
from network.modeling import deeplabv3plus_resnet101

# Load the model with pretrained weights
model = deeplabv3plus_resnet101(num_classes=19, output_stride=8)
model.load_state_dict(torch.load("best_deeplabv3plus_resnet101_cityscapes_os16.pth")['model_state'])
model.eval()  # Set the model to evaluation mode
print("Model loaded successfully!")
