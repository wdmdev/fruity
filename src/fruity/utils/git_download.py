import requests
import os
from tqdm import tqdm


def download_github_folder(repo_url: str, branch: str, folder_path: str, target_dir: str) -> None:
    """Downloads a specific folder from a GitHub repository as a zip file with a progress bar including ETA.

    Args:
    ----
        repo_url (str): URL of the GitHub repository.
        branch (str): Branch of the GitHub repository.
        folder_path (str): Path to the folder in the GitHub repository.
        target_dir (str): Path to the target directory where the folder will be downloaded.
    """
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Construct the URL to the archive (zip) of the folder
    archive_url = f"{repo_url}/archive/{branch}.zip"

    # Make the HTTP request
    response = requests.get(archive_url, stream=True)
    if response.status_code == 200:
        total_size_in_bytes = int(response.headers.get("content-length", 0))
        block_size = 1024  # 1 Kibibyte

        with tqdm(
            total=total_size_in_bytes, unit="iB", unit_scale=True, desc="Downloading Dataset Fruits-360"
        ) as progress_bar:
            zip_path = os.path.join(target_dir, f"{folder_path}.zip")
            with open(zip_path, "wb") as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong")
            print(f"Folder downloaded as zip file: {zip_path}")
    else:
        print(f"Failed to download folder. Status code: {response.status_code}")
