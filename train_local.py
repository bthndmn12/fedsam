from transformers import SamProcessor
from torch.utils.data import DataLoader
# from utils.datalaoder import SAMDataset, DataLoaderdFromDataset
from utils.dataloader_local import WaterDatasetLoader, SAMDataset
from transformers import SamModel
from torch.optim import Adam
from monai.losses import DiceCELoss
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from statistics import mean
import torch
from torch.nn.functional import threshold, normalize
import torch.nn as nn



class TrainModel:

    def __init__(self, dataset_root, image_subfolder, annotation_subfolder):

        self.dataset_root = dataset_root
        self.image_subfolder = image_subfolder
        self.annotation_subfolder = annotation_subfolder
        
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

        self.model = nn.DataParallel(self.model)
        self.model.to(device)
    
    def train(self):
        """Train the model.
        """
        # Load the dataset
        loader = WaterDatasetLoader(self.dataset_root, self.image_subfolder, self.annotation_subfolder)
        loader.load_paths()
        train_data, test_data = loader.create_dataset()
        train_dataset = SAMDataset(dataset=train_data, processor=self.processor)
        test_dataset = SAMDataset(dataset=test_data, processor=self.processor)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        # train_dataset = SAMDataset(dataset=train_dataloader, processor=self.processor)
        # test_dataset = SAMDataset(dataset=test_dataloader, processor=self.processor)
        # train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

        # Load the model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

        # make sure we only compute gradients for mask decoder
        for name, param in model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)

                # initialize the optimizer and the loss function
        optimizer = Adam(self.model.module.mask_decoder.parameters(), lr=1e-5, weight_decay=0)
        
        # we use the DiceCELoss from MONAI because it is efficient in segmentation tasks also HF implementation uses that loss
        seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')
        scaler = GradScaler()

        # Training loop
        

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model= nn.DataParallel(model)
        model.to(device)
        model.train()
        scaler = GradScaler()

        for epoch in range(10):
            epoch_losses = []
            for batch in tqdm(train_dataloader):
                outputs = model(pixel_values=batch["pixel_values"].to(device),
                                input_boxes=batch["input_boxes"].to(device),
                                multimask_output=False)

                predicted_masks = outputs.pred_masks.squeeze(1)
                ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))

                if torch.isfinite(loss):
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_losses.append(loss.item())
                else:
                    print(f'Skipping a step due to non-finite loss: {loss.item()}')

    
            print(f'EPOCH: {epoch}')
            print(f'Mean loss: {mean(epoch_losses)}')

            # Validation loop
            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in tqdm(test_dataloader):
                    outputs = model(pixel_values=batch["pixel_values"].to(device),
                                    input_boxes=batch["input_boxes"].to(device),
                                    multimask_output=False)

                    predicted_masks = outputs.pred_masks.squeeze(1)
                    ground_truth_masks = batch["ground_truth_mask"].float().to(device)
                    loss = seg_loss(predicted_masks, ground_truth_masks.unsqueeze(1))
                    val_losses.append(loss.item())

            print(f'Validation loss: {mean(val_losses)}')


        model.train()

if __name__ == "__main__":
    dataset_root = "D:\\fedsam\\fedsamv1\\images"
    image_subfolder = "JPEGImages\ADE20K"
    annotation_subfolder = "Annotations\ADE20K"

    train_model = TrainModel(dataset_root, image_subfolder, annotation_subfolder)
    train_model.train()