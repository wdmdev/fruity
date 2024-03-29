"""Module for running prediction on a given model and dataloader."""
import torch
import timm
from fruity.datamodules.fruits360 import Fruits360


from torchvision import transforms
from torch.utils.data import DataLoader


@torch.no_grad()
def main(model_path: str, dataset_path: str) -> None:
    """Quantize a model and tests that the performance is the same.

    Args:
    ----
        model_path: path to model
        dataset_path: path to dataset

    Returns:
    -------
        nothing - only prints

    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = Fruits360(dataset_path, train=True, transform=transform)

    state_dict = torch.load(model_path, map_location=device)
    model = timm.models.create_model("resnet18", pretrained=False, in_chans=3, num_classes=131)
    model.load_state_dict(state_dict)
    model.eval()

    dataloader = DataLoader(dataset, batch_size=200)

    for img, label in dataloader:
        preds = model(img).argmax(dim=1)
        print(preds[:4])
        print((preds == label).to(float).mean())

        model.qconfig = torch.quantization.get_default_qconfig("x86")
        model_prepared = torch.quantization.prepare(model)

        # 2. calibrate using dummy data. You should use samples from your dataset.

        model_prepared(img)

        # 3. Convert
        qmodel_int8 = torch.quantization.convert(model_prepared, inplace=True)

        print(qmodel_int8(img).argmax(dim=1)[:4])
        break


if __name__ == "__main__":
    main(r"modelfoods_101.pth", r"data/raw/foods_101")
