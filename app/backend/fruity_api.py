"""FastAPI application entry point.

This module contains code for a FastAPI application
that serves as the backend for the Fruity application.

The API provides endpoints for the following:
- User authentication
- Fruit classification using a pre-trained TIMM model
"""
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from typing import Mapping
import json

import torch
from torchvision import transforms
import timm
from starlette.responses import RedirectResponse
from google.cloud import storage

# Initialize the FastAPI application
app = FastAPI()


# Load the TIMM model
def load_model() -> timm.models:
    """Load the TIMM model.

    Returns
    -------
        torch.nn.Module: The loaded TIMM model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model from check_point state_dict
    model = timm.models.create_model("resnet18", pretrained=False, in_chans=3, num_classes=131)

    # Create a Cloud Storage client.
    gcs = storage.Client()

    # Get the bucket that the model is stored in.
    bucket = gcs.get_bucket("fruity-model-registry")

    # Get the blob with the model.
    blob = bucket.blob("model.pth")

    # Download the model to a local file.
    blob.download_to_filename("/tmp/model.pth")

    state_dict = torch.load("/tmp/model.pth", map_location=device)
    model.load_state_dict(state_dict)

    # Hack to get the idx_to_class mapping
    with open("idx_to_class.json", "r") as f:
        # load json with key as int and value as string
        mapping = json.load(f)
        # convert key to int
        mapping = {int(k): v for k, v in mapping.items()}
        model.idx_to_class = mapping

    model.preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return model


CLASSIFICATION_MODEL: timm.models = load_model()


# Endpoints
@app.get("/", include_in_schema=False)
async def redirect_to_docs() -> RedirectResponse:
    """Redirect to the API documentation."""
    return RedirectResponse(url="/docs")


# Fruit classification endpoint for classifying a single food image
@app.post("/image/classification")
async def classify_fruit(
    file: UploadFile = File(...),
) -> Mapping[str, str]:
    """Classify a single food image.

    Args:
    ----
        file (UploadFile): The food image to classify.

    Returns:
    -------
        dict: A dictionary containing the classification result.
    """
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        raise ValueError("Invalid file type. Must be png or jpeg.")

    # Load the image
    file.file.seek(0)  # reset file pointer
    img = Image.open(file.file)

    # Classify the image
    img = CLASSIFICATION_MODEL.preprocess(img)
    img = img.unsqueeze(0)  # add batch dimension
    label_id = torch.argmax(CLASSIFICATION_MODEL(img)).item()
    result = CLASSIFICATION_MODEL.idx_to_class[label_id]

    return {"result": result}
