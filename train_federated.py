from transformers import SamProcessor
from torch.utils.data import DataLoader
from utils.datalaoder import SAMDataset, DataLoaderdFromDataset
from transformers import SamModel
from torch.optim import Adam
from monai.losses import DiceCELoss
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
import torch.nn as nn
from collections import OrderedDict

"""ToDO:
    there is no train_dataloader in the TrainModel class
    create a function to load the model from the server."""

class TrainModel:
    """
    Class representing the training model.

    Args:
        dataset_root (str): Root directory of the dataset.
        image_subfolder (str): Subfolder containing the images.
        annotation_subfolder (str): Subfolder containing the annotations.
        batch_size (int, optional): Batch size for training. Defaults to 2.
        num_epochs (int, optional): Number of training epochs. Defaults to 70.
    """

    def __init__(self, dataset_root, image_subfolder, annotation_subfolder, batch_size=2, num_epochs=70):
        """ Constructor method.
        
        Args:
            dataset_root (str): Root directory of the dataset.
            image_subfolder (str): Subfolder containing the images.
            annotation_subfolder (str): Subfolder containing the annotations.
            batch_size (int, optional): Batch size for training. Defaults to 2.
            num_epochs (int, optional): Number of training epochs. Defaults to 70.
            """
  
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

        self.model = nn.DataParallel(self.model)
        self.model.to(device)

        # Other attributes
        self.dataset_root = dataset_root
        self.image_subfolder = image_subfolder
        self.annotation_subfolder = annotation_subfolder
        self.batch_size = batch_size
        self.num_epochs = num_epochs
    
    def set_model_parameters(self, model, parameters):
        """Set model parameters.
            Args:
            model (nn.Module): Model to set the parameters for.
            parameters (list): List of NumPy ndarrays representing the parameters.
            """
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict)

    def get_model_parameters(self, model):
        """Get model parameters as a list of NumPy ndarrays.
        Args:
            model (nn.Module): Model to get the parameters from.
            Returns:
            list: List of NumPy ndarrays representing the parameters."""
        
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def train_dataloader(self):
        """Create the training dataloader.
        Returns:
            DataLoader: Training dataloader."""
        loader = DataLoaderdFromDataset(self.dataset_root, self.image_subfolder, self.annotation_subfolder)
        dataset = loader.create_dataset()
        train_dataset = SAMDataset(dataset=dataset, processor=self.processor)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        return train_dataloader

    def train(self, initial_parameters=None):
        """Train the model.
        Args:
            initial_parameters (list, optional): List of NumPy ndarrays representing the initial parameters. Defaults to None.
            Returns:
            list: List of NumPy ndarrays representing the updated parameters.
            """
        # load initial parameters if provided
        if initial_parameters is not None:
            self.set_model_parameters(self.model, initial_parameters)
            print("Loaded initial model parameters.")

        train_dataloader = self.train_dataloader()
        # loader = DataLoaderdFromDataset(self.dataset_root, self.image_subfolder, self.annotation_subfolder)
        # dataset = loader.create_dataset()
        # train_dataset = SAMDataset(dataset=dataset, processor=self.processor)
        # train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        # initialize the optimizer and the loss function
        optimizer = Adam(self.model.module.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
        # we use the DiceCELoss from MONAI because it is efficient in segmentation tasks also HF implementation uses that loss
        seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        scaler = GradScaler()

        self.model.train()
        
        # Get the device
        device = next(self.model.parameters()).device

        # training part
        for epoch in range(self.num_epochs):
            epoch_losses = []
            for batch in tqdm(train_dataloader):
                with autocast():
                    outputs = self.model(pixel_values=batch["pixel_values"].to(device),
                                         input_boxes=batch["input_boxes"].to(device),
                                         multimask_output=False)
                    predicted_masks = outputs.pred_masks.squeeze(1)
                    ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                    loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_losses.append(loss.item())

            print(f'EPOCH: {epoch}')
            print(f'Mean loss: {mean(epoch_losses)}')

        return self.get_model_parameters(self.model)



if __name__ == "__main__":
    
    dataset_root = "D:\\fedsam\\water_v1"
    image_subfolder = "JPEGImages\ADE20K" 
    annotation_subfolder = "Annotations\ADE20K"

    train_model = TrainModel(dataset_root, image_subfolder, annotation_subfolder)
    train_model.train()
