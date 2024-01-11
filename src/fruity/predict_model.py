"""Module for running prediction on a given model and dataloader."""
import torch
from torchvision.datasets import ImageFolder
import timm
from fruity.datamodules.fruits360 import Fruits360

import click
from torchvision import transforms
from torch.utils.data import DataLoader


@click.command()
@click.option("--top_k", default=1, help="Number of top most likely classes to return.")
@click.option("--ckpt_path", default="models/fruity1.ckpt", help="Path to the model checkpoint.")
@click.option("--data_dir", default="data/raw/predict_fruits", help="Directory containing the images.")
@click.option("--batch_size", default=1, help="Batch size for the dataloader.")
def main(top_k: int, ckpt_path: str, data_dir: str, batch_size: int) -> None:
    """Run prediction for a given model and dataloader.

    Args:
    ----
        top_k: Number of top most likely classes to return
        ckpt_path: Path to the model checkpoint
        data_dir: Directory containing the images
        batch_size: Batch size for the dataloader

    Returns:
    -------
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = Fruits360("data/raw/fruits_360", train=True)

    # load model from check_point state_dict
    check_point = torch.load(ckpt_path, map_location=device)
    state_dict = check_point["state_dict"]
    # Remove 'net.' prefix in state_dict
    state_dict = {k.replace("net.", ""): v for k, v in state_dict.items()}
    model = timm.models.create_model("resnet18", pretrained=True, in_chans=3, num_classes=131)
    model.load_state_dict(state_dict)

    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    with torch.no_grad():
        # Predict top 3 most likely class for each image
        for imgs, _ in dataloader:
            preds = model(imgs)
            preds = torch.topk(preds, top_k).indices.squeeze(0).tolist()
            preds = [train_dataset.idx_to_class[pred] for pred in preds]
            print(preds)


if __name__ == "__main__":
    main()
