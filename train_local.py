from transformers import SamProcessor, SamModel
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from torch.optim import Adam
from monai.losses import DiceCELoss
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from statistics import mean
from utils.dataloader_local import WaterDatasetLoader, SAMDataset

class SAMTrainingPipeline:
    def __init__(self, dataset_root, image_subfolder, annotation_subfolder, pretrained_model="facebook/sam-vit-base", batch_size=20, learning_rate=1e-5, num_epochs=100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = SamProcessor.from_pretrained(pretrained_model)
        self.model = SamModel.from_pretrained(pretrained_model).to(self.device)
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.dataset_root = dataset_root
        self.image_subfolder = image_subfolder
        self.annotation_subfolder = annotation_subfolder

        self._prepare_dataloaders()
        self._setup_model()
        self._initialize_optimizer_and_loss()

    def _prepare_dataloaders(self):
        loader = WaterDatasetLoader(self.dataset_root, self.image_subfolder, self.annotation_subfolder)
        loader.load_paths()
        train_data, test_data = loader.create_dataset()
        train_dataset = SAMDataset(dataset=train_data, processor=self.processor)
        test_dataset = SAMDataset(dataset=test_data, processor=self.processor)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def _setup_model(self):
        for name, param in self.model.named_parameters():
            if name.startswith("vision_encoder") or name.startswith("prompt_encoder"):
                param.requires_grad_(False)
        self.model = nn.DataParallel(self.model).to(self.device)

    def _initialize_optimizer_and_loss(self):
        self.optimizer = Adam(self.model.module.mask_decoder.parameters(), lr=self.learning_rate, weight_decay=0)
        self.seg_loss = DiceCELoss(sigmoid=True, squared_pred=True, reduction='mean')

    def train(self):
        scaler = GradScaler()
        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_losses = []
            for batch in tqdm(self.train_dataloader):
                self._train_batch(batch, epoch_losses, scaler)
            print(f'EPOCH: {epoch}')
            print(f'Mean loss: {mean(epoch_losses)}')

            self._validate()

    def _train_batch(self, batch, epoch_losses, scaler):
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



dataset_root = "D:\\fedsam\\water_v1"
image_subfolder = "JPEGImages\ADE20K"
annotation_subfolder = "Annotations\ADE20K"

pipeline = SAMTrainingPipeline(dataset_root, image_subfolder, annotation_subfolder)
pipeline.train()
