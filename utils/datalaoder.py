from datasets import Dataset as Dataset
from torch.utils.data import Dataset as DatasetTorch
from PIL import Image
from utils.bounding_box import get_bounding_box
import numpy as np
import os

class DataLoaderdFromDataset:
    """Class for loading and preprocessing images and annotations"""
    def __init__(self, dataset_root, image_subfolder, annotation_subfolder):
        """ DataLoader class constructor
        
        Args:
            dataset_root (str): Root directory of the dataset
            image_subfolder (str): Name of the image subfolder
            annotation_subfolder (str): Name of the annotation subfolder
            """
        self.dataset_root = dataset_root
        self.image_subfolder = image_subfolder
        self.annotation_subfolder = annotation_subfolder

    def create_dataset(self):
        """Create a dataset using the images and annotations in the dataset root directory
        Returns:
            dataset (datasets.Dataset): A dataset containing the images and annotations"""
        
        # Create lists to store image and annotation file paths
        image_paths = []
        annotation_paths = []

        # Traverse the dataset directory and its subfolders
        for root, dirs, files in os.walk(self.dataset_root):
            # Check if the current directory is an image subfolder
            if self.image_subfolder in root:
                # Iterate through image files
                for filename in files:
                    if filename.endswith(".png"):
                        image_path = os.path.join(root, filename)
                        # Create the corresponding annotation path
                        annotation_path = image_path.replace(self.image_subfolder, self.annotation_subfolder)
                        annotation_path = annotation_path.replace(".png", ".png")

                        # Append the paths to the lists
                        if os.path.exists(annotation_path):  # Check if annotation file exists
                            image_paths.append(image_path)
                            annotation_paths.append(annotation_path)
                        else:
                            print(f"Warning: Annotation file not found for image {image_path}")

        # Verify if the lengths of image_paths and annotation_paths are the same
        if len(image_paths) != len(annotation_paths):
            print("Warning: Mismatch between the number of images and annotations.")

        # Create lists to store image and annotation data
        images = []
        annotations = []

        # Load and preprocess images
        for image_path in image_paths:
            img = self.load_and_preprocess_image(image_path)
            images.append(img)

        # Load and preprocess annotations
        for annotation_path in annotation_paths:
            ann = self.load_and_preprocess_annotation(annotation_path)
            annotations.append(ann)
        
        images = np.array(images, dtype=np.uint8)
        annotations = np.array(annotations, dtype=np.uint8)

        # Convert the NumPy arrays to Pillow images and store them in a dictionary
        dataset_dict = {
        "image": [Image.fromarray(img, 'RGB') for img in images],
        "label": [Image.fromarray(ann, 'RGB') for ann in annotations],  # Keep annotations as grayscale
    }

        # Create the dataset using the datasets.Dataset class
        dataset = Dataset.from_dict(dataset_dict)
        

        # Assuming you have 'images' and 'annotations' as (2167, 256, 256, 3) arrays
        # You can remove the last dimension (3) like this:

        images = images[:, :, :, 0]  # This keeps only the red channel
        annotations = annotations[:, :, :, 0]  # This keeps only the red channel
        # Convert the NumPy arrays to Pillow images and store them in a dictionary
        dataset_dict = {
            "image": [Image.fromarray(img,'L') for img in images],
            "label": [Image.fromarray(ann,'L') for ann in annotations],
        }

        # Create the dataset using the datasets.Dataset class
        dataset = Dataset.from_dict(dataset_dict)
        # Now 'images' and 'annotations' will be (2167, 256, 256) arrays.
        return dataset


    def load_and_preprocess_image(self, image_path, target_size=(256, 256), dtype=np.uint8):
        """Load an image from a file and preprocess it
        Args:
            image_path (str): Path to the image file
            target_size (tuple): Target size of the image
            dtype (type): Data type to use for the loaded image
            Returns:
                img (np.array): Processed image"""
        

        img = Image.open(image_path)  # Convert image to grayscale
        img = img.resize(target_size)  # Resize image to target size
        img = np.array(img, dtype=dtype)
        return img
    
    def load_and_preprocess_annotation(self, annotation_path, target_size=(256, 256), dtype=np.uint8):
        """Load an annotation from a file and preprocess it
        Args:
            annotation_path (str): Path to the annotation file
            target_size (tuple): Target size of the annotation
            dtype (type): Data type to use for the loaded annotation
            Returns:
                ann (np.array): Processed annotation"""
        

        ann = Image.open(annotation_path) # Convert annotation to grayscale
        ann = ann.resize(target_size)
        ann = np.array(ann, dtype=dtype) 
        return ann


class SAMDataset(DatasetTorch):
    
    """
    This class is used to create a dataset that serves input images and masks.
    It takes a dataset and a processor as input and overrides the __len__ and __getitem__ methods of the Dataset class.
    """
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
    def __len__(self):
        """Get the length of the dataset
        Returns:
            length (int): Length of the dataset"""
        return len(self.dataset)

    def __getitem__(self, idx):
        """Get an item from the dataset at the given index
        Args:
            idx (int): Index of the item to get
            Returns:
                inputs (dict): Dictionary containing the inputs for the model"""
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])
        prompt = get_bounding_box(ground_truth_mask)
        inputs = self.processor(image,input_boxes=[[prompt]], return_tensors="pt")
    # remove batch dimension which the processor adds by default
        inputs = {k:v.squeeze(0) for k,v in inputs.items()}

    # add ground truth segmentation
        inputs["ground_truth_mask"] = ground_truth_mask

        return inputs
       
