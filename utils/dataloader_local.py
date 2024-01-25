import os
import numpy as np
from PIL import Image
from datasets import Dataset
from torch.utils.data import Dataset as DatasetTorch
from utils.bounding_box import get_bounding_box
from sklearn.model_selection import train_test_split




class WaterDatasetLoader:
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
        self.image_paths = []  # Initialize as instance attributes
        self.annotation_paths = []  # Initialize as instance attributes

    def load_paths(self):
        for root, dirs, files in os.walk(self.dataset_root):
            if self.image_subfolder in root:
                for filename in files:
                    if filename.endswith(".png"):
                        image_path = os.path.join(root, filename)
                        annotation_path = image_path.replace(self.image_subfolder, self.annotation_subfolder)
                        annotation_path = annotation_path.replace(".png", ".png")

                        if os.path.exists(annotation_path):
                            self.image_paths.append(image_path)
                            self.annotation_paths.append(annotation_path)
                        else:
                            print(f"Warning: Annotation file not found for image {image_path}")

        if len(self.image_paths) != len(self.annotation_paths):
            print("Warning: Mismatch between the number of images and annotations.")

    @staticmethod
    def load_and_preprocess_image(image_path, target_size=(256, 256), dtype=np.uint8):
        img = Image.open(image_path)
        img = img.resize(target_size)
        img = np.array(img, dtype=dtype)
        return img

    @staticmethod
    def load_and_preprocess_annotation(annotation_path, target_size=(256, 256), dtype=np.uint8):
        ann = Image.open(annotation_path)
        ann = ann.resize(target_size)
        ann = np.array(ann, dtype=dtype)/255 
        return ann

    def create_dataset(self):
        images = [self.load_and_preprocess_image(path) for path in self.image_paths]
        annotations = [self.load_and_preprocess_annotation(path) for path in self.annotation_paths]
    
        print("Number of loaded images:", len(images))
        print("Number of loaded annotations:", len(annotations))
        if images and annotations:  # Check if lists are not empty
            print("Shape of first image:", np.array(images[0]).shape)
            print("Shape of first annotation:", np.array(annotations[0]).shape)
    
        images = np.array(images)
        annotations = np.array(annotations)
    
        if images.ndim == 4 and annotations.ndim == 4:
            images = images[:, :, :, 0]
            annotations = annotations[:, :, :, 0]
        else:
            raise ValueError("Expected images and annotations to be 4-dimensional")
    
        # dataset_dict = {
        #     "image": [Image.fromarray(img, 'L') for img in images],
        #     "label": [Image.fromarray(ann, 'L') for ann in annotations],
        # }

        images_train, images_test, annotations_train, annotations_test = train_test_split(images, annotations, test_size=0.2, random_state=42)
        train_dataset_dict = { 
            "image": [Image.fromarray(img, 'L') for img in images_train],
            "label": [Image.fromarray(ann, 'L') for ann in annotations_train],
        }
        test_dataset_dict = {
            "image": [Image.fromarray(img, 'L') for img in images_test],
            "label": [Image.fromarray(ann, 'L') for ann in annotations_test],
        }
        train_dataset = Dataset.from_dict(train_dataset_dict)
        test_dataset = Dataset.from_dict(test_dataset_dict)


        return train_dataset, test_dataset

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