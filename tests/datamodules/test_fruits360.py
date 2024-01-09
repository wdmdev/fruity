import os
from unittest.mock import patch
from torchvision.transforms import transforms
from fruity.datamodules.fruits360 import Fruits360DataModule, Fruits360

class TestFruits360DataModule:
    def test_init(self):
        # Arrange
        data_dir = os.path.join("data", "raw", "fruits_360")
        train_val_test_split = (42_692, 25_000, 22_688)
        batch_size = 64
        num_workers = 0
        pin_memory = False

        # Act
        datamodule = Fruits360DataModule(data_dir, train_val_test_split, batch_size, num_workers, pin_memory)

        # Assert
        assert datamodule.hparams["data_dir"] == data_dir
        assert datamodule.hparams["train_val_test_split"] == train_val_test_split
        assert datamodule.hparams["batch_size"] == batch_size
        assert datamodule.hparams["num_workers"] == num_workers
        assert datamodule.hparams["pin_memory"] == pin_memory
        assert datamodule.data_train is None
        assert datamodule.data_val is None
        assert datamodule.data_test is None

    def test_num_classes(self):
        # Arrange
        datamodule = Fruits360DataModule()

        # Act
        num_classes = datamodule.num_classes

        # Assert
        assert num_classes == 131

    @patch("fruity.datamodules.fruits360.Fruits360DataModule.prepare_data")
    def test_prepare_data(self, mock_prepare_data):
        # Arrange
        datamodule = Fruits360DataModule()

        # Act
        datamodule.prepare_data()

        # Assert
        mock_prepare_data.assert_called_once()


class TestFruits360Dataset:
    def test_image_size(self):
        # Arrange
        root_dir = os.path.join("data", "raw", "fruits_360")
        train = True
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        dataset = Fruits360(root_dir, train, transform)

        # Act
        image, _ = dataset[0]
        image_size = image.shape[1:] # image.shape[0] is the color channel

        # Assert
        assert image_size == (100, 100)

