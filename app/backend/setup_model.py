"""Download model from google cloud storage."""
from google.cloud import storage


if __name__ == "__main__":
    # Initialize a client
    client = storage.Client()

    # Get the bucket
    bucket = client.get_bucket("fruity-model-registry")

    # Get the blob
    blob = bucket.get_blob("model.pth")

    # Download the blob
    blob.download_to_filename("models/model.pth")
