import os
import numpy as np
from PIL import Image
from datasets import Dataset
from torch.utils.data import Dataset as TorchDataset
from sklearn.model_selection import train_test_split
from utils.bounding_box import get_bounding_box
from torchvision import transforms
import logging

class EnhancedWaterDatasetLoader:
    """
    Enhanced loader and preprocessor for a water images dataset, incorporating advanced error handling,
    logging with different levels, and data augmentation.
    """

    def __init__(self, dataset_root, image_subfolder, annotation_subfolder):
        self.dataset_root = dataset_root
        self.image_subfolder = image_subfolder
        self.annotation_subfolder = annotation_subfolder
        self.image_paths = []
        self.annotation_paths = []
        # Initialize logging with a more detailed format
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        self.load_paths()

    def load_paths(self):
        for root, dirs, files in os.walk(self.dataset_root):
            if self.image_subfolder in root:
                for filename in files:
                    if filename.endswith(".png"):
                        image_path = os.path.join(root, filename)
                        annotation_path = image_path.replace(self.image_subfolder, self.annotation_subfolder).replace(".png", ".png")
                        if not os.path.exists(annotation_path):
                            logging.warning(f"Annotation file not found for image {image_path}")
                            continue
                        self.image_paths.append(image_path)
                        self.annotation_paths.append(annotation_path)
        if len(self.image_paths) != len(self.annotation_paths):
            logging.error("Mismatch between the number of images and annotations detected.")
            raise ValueError("Mismatch between the number of images and annotations detected.")

    @staticmethod
    def load_and_preprocess_image(image_path, target_size=(256, 256), dtype=np.uint8):
        try:
            img = Image.open(image_path).resize(target_size)
            img = np.array(img, dtype=dtype)
            return img
        except FileNotFoundError:
            logging.error(f"Image file not found: {image_path}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading image {image_path}: {e}")
            raise

    @staticmethod
    def load_and_preprocess_annotation(annotation_path, target_size=(256, 256), dtype=np.uint8):
        try:
            ann = Image.open(annotation_path).resize(target_size)
            ann = np.array(ann, dtype=dtype) / 255
            return ann
        except FileNotFoundError:
            logging.error(f"Annotation file not found: {annotation_path}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error loading annotation {annotation_path}: {e}")
            raise

    def create_dataset(self, augment=False):
        images = [self.load_and_preprocess_image(path) for path in self.image_paths]
        annotations = [self.load_and_preprocess_annotation(path) for path in self.annotation_paths]
        images_train, images_test, annotations_train, annotations_test = train_test_split(images, annotations, test_size=0.2, random_state=42)

        if augment:
            augmentation_transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                # Add more transforms as needed
            ])
            images_train = [augmentation_transforms(Image.fromarray(img)) for img in images_train]

        train_dataset_dict = {
            "image": images_train,
            "label": annotations_train,
        }
        test_dataset_dict = {
            "image": images_test,
            "label": annotations_test,
        }

        train_dataset = Dataset.from_dict(train_dataset_dict)
        test_dataset = Dataset.from_dict(test_dataset_dict)
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
