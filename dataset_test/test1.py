import numpy as np
import matplotlib.pyplot as plt
import random
import sys
sys.path.append('../')
from utils.dataloader_local import WaterDatasetLoader

def plot_image_and_mask(dataset):

    """
    Plot three random images from the dataset and their corresponding masks
    Args:
        dataset (list): List containing the images and their corresponding masks
        
    Returns:
        None
    """
    # Get three random images from the dataset
    images = np.array([np.array(data["image"]) for data in dataset])
    if images.shape[0] == 0:
        print("No images in the dataset")
        return
    img_nums = random.sample(range(images.shape[0]), 3)
    example_images = [dataset[img_num]["image"] for img_num in img_nums]
    example_masks = [dataset[img_num]["label"] for img_num in img_nums]

    fig, axes = plt.subplots(3, 2, figsize=(10, 15))

    for i in range(3):
        # Plot the image on the left
        axes[i, 0].imshow(np.array(example_images[i]), cmap='gray')  # Assuming the image is grayscale
        axes[i, 0].set_title("Image {}".format(i+1))

        # Plot the mask on the right
        axes[i, 1].imshow(example_masks[i], cmap='gray')  # Assuming the mask is grayscale
        axes[i, 1].set_title("Mask {}".format(i+1))

        # Hide axis ticks and labels
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        axes[i, 0].set_xticklabels([])
        axes[i, 0].set_yticklabels([])
        axes[i, 1].set_xticks([])
        axes[i, 1].set_yticks([])
        axes[i, 1].set_xticklabels([])
        axes[i, 1].set_yticklabels([])

    # Display the images and masks
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
     
    # Define the root directory of your dataset
     dataset_root = "D:\\fedsam\\water_v1"
     image_subfolder = "JPEGImages\ADE20K"
     annotation_subfolder = "Annotations\ADE20K"

    # Create a DataLoader object
     loader = WaterDatasetLoader(dataset_root, image_subfolder, annotation_subfolder)
     loader.load_paths()
     train_data,test_data = loader.create_dataset()
     

     print(train_data)
     plot_image_and_mask(train_data)
    
