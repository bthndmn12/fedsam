import flwr as fl
import sys
sys.path.append('../')
from fedsamv1.train_federated_improved import TrainFederated

class SAMClient(fl.client.NumPyClient):
    """
    Flower client implementing SAM
    
    Args:
        dataset_root (str): Root directory of the dataset
        image_subfolder (str): Name of the image subfolder
        annotation_subfolder (str): Name of the annotation subfolder
        batch_size (int): Batch size for training
        num_epochs (int): Number of epochs for training

    ToDos:
    - Validate the parameters and return types
    """

    def __init__(self, dataset_root, image_subfolder, annotation_subfolder, batch_size=2, num_epochs=2):
        self.train_model = TrainFederated(dataset_root, image_subfolder, annotation_subfolder, batch_size, num_epochs)

    def get_parameters(self, **kwargs):
        # returns initial parameters (before training)
        return self.train_model.get_model_parameters(self.train_model.model)

    def set_parameters(self, parameters):
        # set model parameters received from the server
        self.train_model.set_model_parameters(self.train_model.model, parameters)

    def fit(self, parameters, config):
        # trains the model with the parameters received from the server
        updated_parameters = self.train_model.train(initial_parameters=parameters)
        return updated_parameters, len(self.train_model.train_dataloader().dataset), {}


if __name__ == "__main__":
    dataset_root = "D:\\fedsam\\water_v1"
    image_subfolder = "JPEGImages\\ADE20K"
    annotation_subfolder = "Annotations\\ADE20K"


    client = SAMClient(dataset_root, image_subfolder, annotation_subfolder)
    
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)



# """import flwr as fl
# from collections import OrderedDict   

# class IMDBClient(fl.client.NumPyClient):
#         def get_parameters(self, config):
#             return [val.cpu().numpy() for _, val in net.state_dict().items()]
#         def set_parameters(self, parameters):
#             params_dict = zip(net.state_dict().keys(), parameters)
#             state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#             net.load_state_dict(state_dict, strict=True)
#         def fit(self, parameters, config):
#             self.set_parameters(parameters)
#             print("Training Started...")
#             train(net, trainloader, epochs=1)
#             print("Training Finished.")
#             return self.get_parameters(config={}), len(trainloader), {}
#         def evaluate(self, parameters, config):
#             self.set_parameters(parameters)
#             loss, accuracy = test(net, testloader)
#             return float(loss), len(testloader), {"accuracy": float(accuracy)}


# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=IMDBClient()
# )"""

# import flwr as fl
# from collections import OrderedDict
# from transformers import SamProcessor, SamModel
# from torch.utils.data import DataLoader
# import sys
# sys.path.append('../')
# from utils.datalaoder import SAMDataset, DataLoaderdFromDataset
# from monai.losses import DiceCELoss
# from torch.cuda.amp import autocast, GradScaler
# from tqdm import tqdm
# from statistics import mean
# import torch
# import torch.nn as nn
# from torch.optim import Adam

# # Define your model architecture and training functions
# class SegmentationModel(nn.Module):
#     def __init__(self):
#         super(SegmentationModel, self).__init__()
#         self.model = SamModel.from_pretrained("facebook/sam-vit-base")
#         for name, param in self.model.named_parameters():
#             if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
#                 param.requires_grad_(False)
#         self.mask_decoder = self.model.mask_decoder

#     def forward(self, pixel_values, input_boxes):
#         return self.mask_decoder(pixel_values=pixel_values, input_boxes=input_boxes, multimask_output=False)

# # Define your training loop
# def train(model, train_dataloader, optimizer, seg_loss, scaler, device):
#     model.train()
#     epoch_losses = []
#     for batch in tqdm(train_dataloader):
#         with autocast():
#             outputs = model(pixel_values=batch["pixel_values"].to(device),
#                             input_boxes=batch["input_boxes"].to(device),
#                             multimask_output=False)
#             predicted_masks = outputs.pred_masks.squeeze(1)
#             ground_truth_masks = batch["ground_truth_mask"].float().to(device)
#             loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

#         optimizer.zero_grad()
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         epoch_losses.append(loss.item())

#     return mean(epoch_losses)

# # Federated Learning Client Class
# class SegmentationClient(fl.client.NumPyClient):
#     """Flower client implementing SAM
    
#     Args:
#         model (torch.nn.Module): PyTorch module implementing the model
#         train_dataloader (torch.utils.data.DataLoader): Training data
#         optimizer (torch.optim.Optimizer): Optimizer used to train the model
#         loss_fn (torch.nn.Module): Loss function used to train the model
#         device (torch.device): Device on which the model should be trained
        
#         Returns:
#             parameters (list): Updated model parameters
#             num_examples (int): Number of examples used for training
#             metrics (dict): Dictionary containing the loss and accuracy of the model
#         """
#     def __init__(self, model, train_dataloader, optimizer, seg_loss, scaler, device):
#         self.model = model
#         self.train_dataloader = train_dataloader
#         self.optimizer = optimizer
#         self.seg_loss = seg_loss
#         self.scaler = scaler
#         self.device = device

#     def get_parameters(self, config):
#         return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

#     def set_parameters(self, parameters):
#         params_dict = zip(self.model.state_dict().keys(), parameters)
#         state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
#         self.model.load_state_dict(state_dict, strict=True)

#     def fit(self, parameters, config):
#         self.set_parameters(parameters)
#         print("Training Started...")
#         mean_loss = train(self.model, self.train_dataloader, self.optimizer, self.seg_loss, self.scaler, self.device)
#         print("Training Finished.")
#         return self.get_parameters(config={}), len(self.train_dataloader), {"mean_loss": mean_loss}

#     def evaluate(self, parameters, config):
#         self.set_parameters(parameters)
#         mean_loss = train(self.model, self.train_dataloader, self.optimizer, self.seg_loss, self.scaler, self.device)
#         return mean_loss, len(self.train_dataloader), {"mean_loss": mean_loss}

# # Initialize SAM Processor
# processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

# # Create DataLoader and SAMDataset
# dataset_root = "D:\\fedsam\\water_v1"
# image_subfolder = "JPEGImages\\ADE20K"
# annotation_subfolder = "Annotations\\ADE20K"
# loader = DataLoaderdFromDataset(dataset_root, image_subfolder, annotation_subfolder)
# dataset = loader.create_dataset()
# train_dataset = SAMDataset(dataset=dataset, processor=processor)
# train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=False)

# # Initialize SAM Model, optimizer, and loss function
# model = SegmentationModel()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = nn.DataParallel(model)
# model.to(device)

# optimizer = Adam(model.module.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
# seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

# # Training loop
# num_epochs = 70
# scaler = GradScaler()

# # Federated Learning
# fl.client.start_numpy_client(
#     server_address="127.0.0.1:8080",
#     client=SegmentationClient(model, train_dataloader, optimizer, seg_loss, scaler, device)
# )
