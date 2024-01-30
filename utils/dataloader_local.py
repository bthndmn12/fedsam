import os
import numpy as np
from PIL import Image
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from utils.bounding_box import get_bounding_box

class WaterDatasetLoader:

    """
    A class used to load and preprocess a dataset of water images and their corresponding annotations.

    ...

    Attributes
    ----------
    dataset_root : str
        The root directory of the dataset
    image_subfolder : str
        The subfolder within the root directory that contains the images
    annotation_subfolder : str
        The subfolder within the root directory that contains the annotations
    image_paths : list
        A list of paths to the images in the dataset
    annotation_paths : list
        A list of paths to the annotations in the dataset

    Methods
    -------
    load_paths():
        Loads the paths of the images and annotations into the respective lists.
    load_and_preprocess_image(image_path, target_size=(256, 256), dtype=np.uint8):
        Loads and preprocesses an image from the given path.
    load_and_preprocess_annotation(annotation_path, target_size=(256, 256), dtype=np.uint8):
        Loads and preprocesses an annotation from the given path.
    create_dataset():
        Creates a dataset from the loaded and preprocessed images and annotations.
    """

    def __init__(self, dataset_root, image_subfolder, annotation_subfolder):
         
        """
        Constructs all the necessary attributes for the WaterDatasetLoader object.

        Parameters
        ----------
            dataset_root : str
                The root directory of the dataset
            image_subfolder : str
                The subfolder within the root directory that contains the images
            annotation_subfolder : str
                The subfolder within the root directory that contains the annotations
        """

        self.dataset_root = dataset_root
        self.image_subfolder = image_subfolder
        self.annotation_subfolder = annotation_subfolder
        self.image_paths = []
        self.annotation_paths = []
        self.load_paths()

    def load_paths(self):
        """
        Loads the paths of the images and annotations into the respective lists.
        """
        for root, dirs, files in os.walk(self.dataset_root):
            if self.image_subfolder in root:
                for filename in files:
                    if filename.endswith(".png"):
                        image_path = os.path.join(root, filename)
                        annotation_path = image_path.replace(self.image_subfolder, self.annotation_subfolder).replace(".png", ".png")
                        if os.path.exists(annotation_path):
                            self.image_paths.append(image_path)
                            self.annotation_paths.append(annotation_path)
                        else:
                            print(f"Warning: Annotation file not found for image {image_path}")
        if len(self.image_paths) != len(self.annotation_paths):
            print("Warning: Mismatch between the number of images and annotations.")

    @staticmethod
    def load_and_preprocess_image(image_path, target_size=(256, 256), dtype=np.uint8):
        """
        Loads and preprocesses an image from the given path.

        Parameters
        ----------
            image_path : str
                The path to the image
            target_size : tuple, optional
                The target size of the image (default is (256, 256))
            dtype : data-type, optional
                The desired data-type for the image (default is np.uint8)

        Returns
        -------
            img : ndarray
                The loaded and preprocessed image
        """
        img = Image.open(image_path)
        img = img.resize(target_size)
        img = np.array(img, dtype=dtype)
        return img

    @staticmethod
    def load_and_preprocess_annotation(annotation_path, target_size=(256, 256), dtype=np.uint8):
        """
        Loads and preprocesses an annotation from the given path.

        Parameters
        ----------
            annotation_path : str
                The path to the annotation
            target_size : tuple, optional
                The target size of the annotation (default is (256, 256))
            dtype : data-type, optional
                The desired data-type for the annotation (default is np.uint8)

        Returns
        -------
            ann : ndarray
                The loaded and preprocessed annotation
        """
        ann = Image.open(annotation_path)
        ann = ann.resize(target_size)
        ann = np.array(ann, dtype=dtype) / 255
        return ann

    def create_dataset(self):
        """
        Creates a dataset from the loaded and preprocessed images and annotations.

        Returns
        -------
            train_dataset : Dataset
                The training dataset
            test_dataset : Dataset
                The testing dataset
        """
        images = [self.load_and_preprocess_image(path) for path in self.image_paths]
        annotations = [self.load_and_preprocess_annotation(path) for path in self.annotation_paths]
        images = np.array(images, dtype=np.uint8)
        annotations = np.array(annotations, dtype=np.uint8)

        # dataset_dict = {
        #     "image": [Image.fromarray(img, 'L') for img in images[:, :, :, 0]],
        #     "label": [Image.fromarray(ann, 'L') for ann in annotations[:, :, :, 0]],
        # }  
        images_train, images_test, annotations_train, annotations_test = train_test_split(images, annotations, test_size=0.2, random_state=42)
        train_dataset_dict = { 
            "image": [Image.fromarray(img, 'RGB') for img in images_train],
            "label": [Image.fromarray(ann, 'L') for ann in annotations_train[:, :, :, 0]],
        }
        test_dataset_dict = {
            "image": [Image.fromarray(img, 'RGB') for img in images_test],
            "label": [Image.fromarray(ann, 'L') for ann in annotations_test[:, :, :, 0]],
        }
        train_dataset = Dataset.from_dict(train_dataset_dict)
        test_dataset = Dataset.from_dict(test_dataset_dict)
        # return Dataset.from_dict(dataset_dict)
        return train_dataset, test_dataset

# Code taken from: https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb
class SAMDataset(TorchDataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"]
        ground_truth_mask = np.array(item["label"])
        prompt = get_bounding_box(ground_truth_mask)
        inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["ground_truth_mask"] = ground_truth_mask
        return inputs
