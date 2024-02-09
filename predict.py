from transformers import SamModel, SamConfig, SamProcessor
import torch
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import tkinter as tk
from tkinter import filedialog


device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model configuration
model_config = SamConfig.from_pretrained("facebook/sam-vit-base")
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# Create an instance of the model architecture with the loaded configuration
my_wase_model = SamModel(config=model_config)
# Update the model by loading the weights from saved file.
my_wase_model.load_state_dict(torch.load("./model.pth"))

# Set the device to cuda if available, otherwise use cpu
device = "cuda" if torch.cuda.is_available() else "cpu"
my_wase_model.to(device)


def predict_image(image_path):

    test_image = Image.open(image_path)
    # test_image = test_image.convert("L")
    # Get the box prompt based on the image size (you may need to adjust this based on your use case)
    prompt = [0, 0, test_image.width, test_image.height]

    # Modify the prompt to be in the expected format
    prompt = [[prompt]]  # Wrap it in an extra list

    # Prepare the image + box prompt for the model
    inputs = processor(test_image, input_boxes=prompt, return_tensors="pt")

    # Move the input tensor to the GPU if it's not already there
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Set the model to evaluation mode
    my_wase_model.eval()

    # Forward pass
    with torch.no_grad():
        outputs = my_wase_model(**inputs, multimask_output=False)

    # Apply sigmoid
    medsam_seg_prob = torch.sigmoid(outputs.pred_masks.squeeze(1))
    # Convert soft mask to hard mask
    medsam_seg_prob = medsam_seg_prob.cpu().numpy().squeeze()
    medsam_seg = (medsam_seg_prob > 0.5).astype(np.uint8)


    fig, axes = plt.subplots(1, 3, figsize=(15, 5))


    axes[0].imshow(np.array(test_image), cmap='gray')
    axes[0].set_title("Image")

    axes[1].imshow(medsam_seg, cmap='gray')
    axes[1].set_title("Mask")

    axes[2].imshow(medsam_seg_prob, cmap='viridis')  
    axes[2].set_title("Probability Map")

    plt.show()

    return test_image, medsam_seg, medsam_seg_prob


def select_images():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select Images", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))
    return file_paths


image_paths = select_images()
predictions = []

for image_path in image_paths:
    prediction = predict_image(image_path)
    predictions.append(prediction)


fig, axes = plt.subplots(len(image_paths), 3, figsize=(15, 5 * len(image_paths)))

for i, (image_path, (test_image, medsam_seg, medsam_seg_prob)) in enumerate(zip(image_paths, predictions)):

    axes[i, 0].imshow(np.array(test_image), cmap='gray')
    axes[i, 0].set_title("Image")

    axes[i, 1].imshow(medsam_seg, cmap='gray')
    axes[i, 1].set_title("Mask")


    axes[i, 2].imshow(medsam_seg_prob, cmap='viridis')  # You can choose a different colormap
    axes[i, 2].set_title("Probability Map")

plt.show()
# # Function to open file dialog and select multiple images
# def select_images():
#     root = tk.Tk()
#     root.withdraw()
#     file_paths = filedialog.askopenfilenames(title="Select Images", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png")))
#     return file_paths

# # Example usage
# image_paths = select_images()
# predictions = []

# for image_path in image_paths:
#     prediction = predict_image(image_path)
#     predictions.append(prediction)

# # Display all images after all predictions are done
# fig, axes = plt.subplots(len(image_paths), 3, figsize=(15, 5 * len(image_paths)))

# for i, (image_path, (test_image, medsam_seg, medsam_seg_prob)) in enumerate(zip(image_paths, predictions)):
#     # Plot the original image on the left
#     axes[i, 0].imshow(np.array(test_image), cmap='gray')
#     axes[i, 0].set_title("Image")

#     # Plot the segmentation mask in the middle
#     axes[i, 1].imshow(medsam_seg, cmap='gray')
#     axes[i, 1].set_title("Mask")

#     # Plot the probability map on the right
#     axes[i, 2].imshow(medsam_seg_prob, cmap='viridis')  # You can choose a different colormap
#     axes[i, 2].set_title("Probability Map")

# plt.show()