import unittest
from unittest.mock import patch, MagicMock
import torch
import sys
sys.path.append('../')
# Assuming the file name is train_local.py and the class is TrainModel
from train_local import TrainModel

class TestTrainModel(unittest.TestCase):

    @patch('train_local.SamProcessor.from_pretrained')
    @patch('train_local.SamModel.from_pretrained')
    @patch('train_local.torch.device')
    def setUp(self, mock_device, mock_model, mock_processor):
        # Mock the device to always be cpu for testing
        mock_device.return_value = torch.device('cpu')

        # Mock the model and processor
        mock_model.return_value = MagicMock()
        mock_processor.return_value = MagicMock()

        self.train_model = TrainModel('dataset_root', 'image_subfolder', 'annotation_subfolder')

    @patch('train_local.DataLoaderdFromDataset')
    @patch('train_local.SAMDataset')
    @patch('train_local.DataLoader')
    @patch('train_local.Adam')
    @patch('train_local.DiceCELoss')
    @patch('train_local.GradScaler')
    @patch('train_local.tqdm')
    def test_train(self, mock_tqdm, mock_grad_scaler, mock_dice_loss, mock_adam, mock_data_loader, mock_sam_dataset, mock_data_loader_from_dataset):
        # Mock the return values
        mock_data_loader_from_dataset.return_value.create_dataloaders.return_value = (MagicMock(), MagicMock())
        mock_sam_dataset.return_value = MagicMock()
        mock_data_loader.return_value = MagicMock()
        mock_adam.return_value = MagicMock()
        mock_dice_loss.return_value = MagicMock()
        mock_grad_scaler.return_value = MagicMock()
        mock_tqdm.return_value = iter([MagicMock()])

        self.train_model.train()

        # Assert that the methods were called
        mock_data_loader_from_dataset.assert_called_once_with('dataset_root', 'image_subfolder', 'annotation_subfolder')
        mock_sam_dataset.assert_called()
        mock_data_loader.assert_called()
        mock_adam.assert_called()
        mock_dice_loss.assert_called()
        mock_grad_scaler.assert_called()
        mock_tqdm.assert_called()

if __name__ == '__main__':
    unittest.main()