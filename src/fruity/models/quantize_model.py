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

        model.qconfig = torch.ao.quantization.get_default_qconfig("x86")

        # Fuse the activations to preceding layers, where applicable.
        # This needs to be done manually depending on the model architecture.
        # Common fusions include `conv + relu` and `conv + batchnorm + relu`
        # model_fp32_fused = torch.ao.quantization.fuse_modules(model, [['conv1']])

        # Prepare the model for static quantization. This inserts observers in
        # the model that will observe activation tensors during calibration.
        model_fp32_prepared = torch.ao.quantization.prepare(model)

        # calibrate the prepared model to determine quantization parameters for activations
        # in a real world setting, the calibration would be done with a representative dataset

        model_fp32_prepared(img)

        # Convert the observed model to a quantized model. This does several things:
        # quantizes the weights, computes and stores the scale and bias value to be
        # used with each activation tensor, and replaces key operators with quantized
        # implementations.
        model_int8 = torch.ao.quantization.convert(model_fp32_prepared)

        # run the model, relevant calculations will happen in int8
        res = model_int8(img)

        print(res.argmax(dim=1)[:4])
        break


if __name__ == "__main__":
    main(r"modelfoods_101.pth", r"data/raw/foods_101")
