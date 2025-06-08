# output the image shape
image_path = "28568.jpg"

from PIL import Image
import numpy as np

# Load image using PIL
img = Image.open(image_path)

# Convert to numpy array to get shape
img_array = np.array(img)

print(f"Image shape: {img_array.shape}")
