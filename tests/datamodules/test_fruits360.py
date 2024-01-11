"""Tests for the Fruits360DataModule class.""" ""
import os
from fruity.datamodules.fruits360 import Fruits360DataModule


class TestFruits360DataModule:
    """Tests for the Fruits360DataModule class."""

    def test_init(self) -> None:
        """Test initialization of Fruits360DataModule."""
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

    def test_num_classes(self) -> None:
        """Test num_classes property."""
        datamodule = Fruits360DataModule()

        # Act
        num_classes = datamodule.num_classes

        # Assert
        assert num_classes == 131
