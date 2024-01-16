"""Fruits360 dataset."""
import os
from typing import Optional, Tuple, Callable, Union

from pytorch_lightning import LightningDataModule
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from PIL import Image

from fruity.utils.git_download import download_github_folder


class Fruits360(Dataset):
    """Fruits360 dataset."""

    def __init__(
        self, root_dir: str, train: bool, transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
    ) -> None:
        """Fruits360 dataset.

        Args:
        ----
            root_dir (string):  Directory with all the images.
            train (bool):       If True, creates dataset from training set, otherwise creates from
                                test set.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        if self.train:
            self.root_dir = os.path.join(self.root_dir, "train")
        else:
            self.root_dir = os.path.join(self.root_dir, "test")

        self.classes = os.listdir(self.root_dir)
        self.classes.sort()
        self.idx_to_class = {i: cls for i, cls in enumerate(self.classes)}
        self.images = []
        self.targets = []
        for i, cls in enumerate(self.classes):
            cls_path = os.path.join(self.root_dir, cls)
            for img in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img)
                self.images.append(img_path)
                self.targets.append(i)

    def __len__(self) -> int:
        """Return length of dataset."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return sample from dataset."""
        img_path = self.images[idx]
        img = Image.open(img_path)
        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target


class Fruits360DataModule(LightningDataModule):
    """LightningDataModule for Kaggle Fruits 360 dataset."""

    def __init__(
        self,
        data_dir: str = os.path.join("data", "raw", "fruits_360"),
        train_val_test_split: Tuple[int, int, int] = (42_692, 25_000, 22_688),
        batch_size: int = 64,
        num_workers: Union[str, int] = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
    ) -> None:
        """LightningDataModule for Kaggle Fruits 360 dataset.

        Args:
        ----
            data_dir (str):                 Directory where data is stored.
            train_val_test_split (tuple):   Tuple of ints with lengths of train, val and test
                                            datasets.
            batch_size (int):               Size of batch.
            num_workers (int | str):        How many subprocesses to use for data loading.
            pin_memory (bool):              Whether to copy tensors into CUDA pinned memory.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        if self.hparams.num_workers == "max":
            self.hparams.num_workers = torch.multiprocessing.cpu_count()

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
    def num_classes(self) -> int:
        """Return number of classes."""
        return 131

    def prepare_data(self) -> None:
        """Download data if needed.

        Do not use it to assign state (self.x = y).
        """
        repo_url = "https://github.com/Horea94/Fruit-Images-Dataset"
        branch = "master"
        folder_path = "fruits_360"  # Example folder path
        target_dir = os.path.join("data", "raw")
        if not os.path.exists(os.path.join(target_dir, folder_path)):
            download_github_folder(repo_url, branch, folder_path, target_dir)

            # unzip data/raw/fruits.zip to data/raw/fruits_360
            os.system(f"unzip -q -o -d data/raw/{folder_path} data/raw/{folder_path}.zip")

            # select Train and Test folders from fruits/Fruit-Images-Dataset-master and move them one layer up
            os.system(f"mv data/raw/{folder_path}/Fruit-Images-Dataset-master/Training data/raw/{folder_path}")
            os.system(f"mv data/raw/{folder_path}/Fruit-Images-Dataset-master/Test data/raw/{folder_path}")

            # remove fruits and fruits.zip
            os.system(f"rm -rf data/raw/{folder_path}/Fruit-Images-Dataset-master")
            os.system(f"rm -rf data/raw/{folder_path}.zip")

            # rename data/raw/fruits_360/Test and Train to data/raw/fruits_360/test and train
            os.system(f"mv data/raw/{folder_path}/Test data/raw/{folder_path}/test")
            os.system(f"mv data/raw/{folder_path}/Training data/raw/{folder_path}/train")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!

        Args:
        ----
            stage (str, optional): Stage can be 'fit' or 'test'. If 'fit', `setup()` will split
                                   data into training and validation. If 'test', `setup()` will
                                   not split data.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            trainset = Fruits360(self.hparams.data_dir, train=True, transform=self.transforms)
            testset = Fruits360(self.hparams.data_dir, train=False, transform=self.transforms)
            dataset = ConcatDataset(datasets=[trainset, testset])
            self.data_train, self.data_val, self.data_test = random_split(
                dataset=dataset, lengths=self.hparams.train_val_test_split
            )

    def train_dataloader(self) -> DataLoader:
        """Return training dataloader."""
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test dataloader."""
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            persistent_workers=self.hparams.persistent_workers,
            shuffle=False,
        )
