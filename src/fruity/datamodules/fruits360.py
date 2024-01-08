import os
from typing import Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

from fruity.utils.git_download import download_github_folder

class Fruits360DataModule(LightningDataModule):
    """LightningDataModule for Kaggle Fruits 360 dataset.

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str = os.path.join("data", "raw", "fruits_360"),
        train_val_test_split: Tuple[int, int, int] = (42_692, 25_000, 22_688),
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    @property
    def num_classes(self):
        return 131

    def prepare_data(self):
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        repo_url = 'https://github.com/Horea94/Fruit-Images-Dataset'
        branch = 'master'
        folder_path = 'fruits_360'  # Example folder path
        target_dir = os.path.join('data', 'raw')
        download_github_folder(repo_url, branch, folder_path, target_dir)

        #unzip data/raw/fruits.zip to data/raw/fruits_360 
        os.system(f"unzip -q -o -d data/raw/{folder_path} data/raw/{folder_path}.zip")
        
        #select Train and Test folders from fruits/Fruit-Images-Dataset-master and move them one layer up 
        os.system(f"mv data/raw/{folder_path}/Fruit-Images-Dataset-master/Training data/raw/{folder_path}")
        os.system(f"mv data/raw/{folder_path}/Fruit-Images-Dataset-master/Test data/raw/{folder_path}")

        #remove fruits and fruits.zip
        os.system(f"rm -rf data/raw/{folder_path}/Fruit-Images-Dataset-master")
        os.system(f"rm -rf data/raw/{folder_path}.zip")

        #rename data/raw/fruits_360/Test and Train to data/raw/fruits_360/test and train
        os.system(f"mv data/raw/{folder_path}/Test data/raw/{folder_path}/test")
        os.system(f"mv data/raw/{folder_path}/Training data/raw/{folder_path}/train")

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = ImageFolder(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = ImageFolder(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset,
                lengths=self.hparams.train_val_test_split,
                generator=torch.Generator().manual_seed(42),
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )