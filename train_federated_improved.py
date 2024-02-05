from torch.utils.data import DataLoader
from utils.dataloader_local import SAMDataset, WaterDatasetLoader
from transformers import SamModel, SamProcessor
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
    There is no train_dataloader in the TrainModel class so I have to create it.
    I have to create a function to load the model from the server."""

class TrainFederated:
    """
    Class representing the training model.

    Args:
        dataset_root (str): Root directory of the dataset.
        image_subfolder (str): Subfolder containing the images.
        annotation_subfolder (str): Subfolder containing the annotations.
        batch_size (int, optional): Batch size for training. Defaults to 2.
        num_epochs (int, optional): Number of training epochs. Defaults to 70.
    """

    def __init__(self, dataset_root, image_subfolder, annotation_subfolder, pretrained_model="facebook/sam-vit-base", batch_size=20, learning_rate=1e-5, num_epochs=100):
        """ Constructor method.
        
        Args:
            dataset_root (str): Root directory of the dataset.
            image_subfolder (str): Subfolder containing the images.
            annotation_subfolder (str): Subfolder containing the annotations.
            batch_size (int, optional): Batch size for training. Defaults to 2.
            num_epochs (int, optional): Number of training epochs. Defaults to 70.
            """
  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        self.model = SamModel.from_pretrained("facebook/sam-vit-base").to(self.device)
        
        # self.model = SamModel.from_pretrained(pretrained_model).to(device)

        # for name, param in self.model.named_parameters():
        #     if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
        #         param.requires_grad_(False)
        # self.model = nn.DataParallel(self.model)
        # self.model.to(device)

        # Other attributes
        self.dataset_root = dataset_root
        self.image_subfolder = image_subfolder
        self.annotation_subfolder = annotation_subfolder
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate

        self._prepare_dataloaders()
        self._setup_model()
        self._initialize_optimizer_and_loss()

    
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
    
    def _prepare_dataloaders(self):
        """
        Prepare the train and test dataloaders.

        This function loads the dataset paths, creates the train and test datasets,
        and initializes the train and test dataloaders.
        """
        loader = WaterDatasetLoader(self.dataset_root, self.image_subfolder, self.annotation_subfolder)
        loader.load_paths()
        train_data, test_data = loader.create_dataset()
        train_dataset = SAMDataset(dataset=train_data, processor=self.processor)
        test_dataset = SAMDataset(dataset=test_data, processor=self.processor)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def _setup_model(self):
        """
        Setup the model for training.

        This function freezes the parameters of the vision_encoder and prompt_encoder,
        and initializes the model as a DataParallel model.
        """
        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)
        self.model = nn.DataParallel(self.model).to(self.device)

    def _initialize_optimizer_and_loss(self):
        """
        Initialize the optimizer and loss function.

        This function initializes the Adam optimizer for the mask_decoder parameters
        and the DiceCELoss as the segmentation loss function.
        """
        self.optimizer = Adam(self.model.module.mask_decoder.parameters(), lr=self.learning_rate, weight_decay=0)
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')


    def train(self, initial_parameters=None):
        """
        Train the SAM model.

        This function performs the training loop for the specified number of epochs.
        It trains the model on the train dataset, prints the mean loss for each epoch,
        and validates the model on the test dataset.
        """


        """Train the model.
        Args:
            initial_parameters (list, optional): List of NumPy ndarrays representing the initial parameters. Defaults to None.
            Returns:
            list: List of NumPy ndarrays representing the updated parameters.
            """
        # load initial parameters if provided
        if initial_parameters is not None:
            self.set_model_parameters(self.model, initial_parameters)


        scaler = GradScaler()
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_losses = []
            for batch in tqdm(self.train_dataloader):
                self._train_batch(batch, epoch_losses, scaler)
            print(f'EPOCH: {epoch}')
            print(f'Mean loss: {mean(epoch_losses)}')

            self._validate()
        return self.get_model_parameters(self.model)

    def _train_batch(self, batch, epoch_losses, scaler):
        """
        Train the model on a single batch.

        Args:
            batch (dict): The batch of data containing pixel_values, input_boxes, and ground_truth_mask.
            epoch_losses (list): The list to store the losses for the current epoch.
            scaler (GradScaler): The gradient scaler for mixed precision training.
        """
        outputs = self.model(pixel_values=batch["pixel_values"].to(self.device),
                             input_boxes=batch["input_boxes"].to(self.device),
                             multimask_output=False)

        predicted_masks = outputs.pred_masks.squeeze(1)
        ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
        loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.module.parameters(), max_norm=1.0)  # Gradient clipping
        self.optimizer.step()
        epoch_losses.append(loss.item())

    def _validate(self):
        """
        Validate the model on the test dataset.

        This function evaluates the model on the test dataset and prints the validation loss.
        """
        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                outputs = self.model(pixel_values=batch["pixel_values"].to(self.device),
                                     input_boxes=batch["input_boxes"].to(self.device),
                                     multimask_output=False)

                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(self.device)
                loss = self.seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
                val_losses.append(loss.item())


"""
Below function is old implementation of the train function and is not used in the current implementation.
"""

    # def train(self, initial_parameters=None):
    #     """Train the model.
    #     Args:
    #         initial_parameters (list, optional): List of NumPy ndarrays representing the initial parameters. Defaults to None.
    #         Returns:
    #         list: List of NumPy ndarrays representing the updated parameters.
    #         """
    #     # load initial parameters if provided
    #     if initial_parameters is not None:
    #         self.set_model_parameters(self.model, initial_parameters)

    
    #     loader = WaterDatasetLoader(self.dataset_root, self.image_subfolder, self.annotation_subfolder)
    #     dataset = loader.create_dataset()
    #     train_dataset = SAMDataset(dataset=dataset, processor=self.processor)
    #     train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    #     # initialize the optimizer and the loss function
    #     optimizer = Adam(self.model.module.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
    #     # we use the DiceCELoss from MONAI because it is efficient in segmentation tasks also HF implementation uses that loss
    #     seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
    #     scaler = GradScaler()

    #     self.model.train()
        
    #     # Get the device
    #     device = next(self.model.parameters()).device

    #     # training part
    #     for epoch in range(self.num_epochs):
    #         epoch_losses = []
    #         for batch in tqdm(train_dataloader):
    #             with autocast():
    #                 outputs = self.model(pixel_values=batch["pixel_values"].to(device),
    #                                      input_boxes=batch["input_boxes"].to(device),
    #                                      multimask_output=False)
    #                 predicted_masks = outputs.pred_masks.squeeze(1)
    #                 ground_truth_masks = batch["ground_truth_mask"].float().to(device)
    #                 loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

    #             optimizer.zero_grad()
    #             scaler.scale(loss).backward()
    #             scaler.step(optimizer)
    #             scaler.update()
    #             epoch_losses.append(loss.item())

    #         print(f'EPOCH: {epoch}')
    #         print(f'Mean loss: {mean(epoch_losses)}')

    #     return self.get_model_parameters(self.model)
