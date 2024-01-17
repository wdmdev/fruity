"""Frontend code for the fruity classification application."""

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import os
import av
import numpy as np
import time
import shutil
import atexit

from streamlit_webrtc import VideoProcessorBase


class VideoTransformer(VideoProcessorBase):
    """VideoTransformer class for processing video frames."""

    frame: np.ndarray = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Receive a video frame, process it, and return it.

        Args:
        ----
            frame (av.VideoFrame): The video frame to process.

        Returns:
        -------
            av.VideoFrame: The processed video frame.
        """
        self.process(frame)
        return frame

    def process(self, frame: av.VideoFrame) -> np.ndarray:
        """Process a video frame.

        Args:
        ----
            frame (av.VideoFrame): The video frame to process.

        Returns:
        -------
            np.ndarray: The processed video frame.
        """
        self.frame = frame.to_ndarray(format="bgr24")
        return self.frame


def main() -> None:
    """Start the frontend."""
    st.title("DEMO: fruity classification")

    # Initialize the images list in the session state if it doesn't exist
    if "images" not in st.session_state:
        st.session_state["images"] = []

    # Start the webcam
    webrtc_ctx = webrtc_streamer(key="example", video_processor_factory=VideoTransformer)

    # Capture an image from the webcam
    if st.button("Capture"):
        if webrtc_ctx.video_processor and webrtc_ctx.video_processor.frame is not None:
            # Convert the numpy array to a PIL Image
            img = webrtc_ctx.video_processor.frame

            # Convert the color format from BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Generate a unique filename for each image
            timestamp = int(time.time())
            current_dir = os.path.dirname(os.path.abspath(__file__))
            snapshots_dir = os.path.join(current_dir, "snapshots")
            image_path = os.path.join(snapshots_dir, f"snapshot_{timestamp}.png")

            # Save the image in RGB format
            cv2.imwrite(image_path, img_rgb)

            # Add the image path to the session state
            st.session_state["images"].append(image_path)

    # Display the images in a grid
    if len(st.session_state["images"]) > 0:
        num_columns = 3
        num_rows = (len(st.session_state["images"]) - 1) // num_columns + 1

        for row in range(num_rows):
            cols = st.columns(num_columns)
            for col in range(num_columns):
                index = row * num_columns + col
                if index < len(st.session_state["images"]):
                    with cols[col]:
                        # Read the image file into a numpy array
                        image = cv2.imread(st.session_state["images"][index], cv2.IMREAD_COLOR)
                        st.image(image)


def process_image(img: np.ndarray, image_label: str) -> str:
    """Process the image with your ML model and display the results.

    Args:
    ----
        img (np.ndarray): The image to process.
        image_label (str): The label for the image.

    Returns:
    -------
        str: The image name.
    """
    # Convert the image from BGR to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Save the image to a temporary directory
    path = os.path.join(os.path.dirname(__file__), "snapshots", image_label + ".png")
    cv2.imwrite(path, rgb)

    # Return the image name
    return path


def clear_snapshots() -> None:
    """Clear the snapshots directory."""
    shutil.rmtree(os.path.join(os.path.dirname(__file__), "snapshots"))
    os.makedirs(os.path.join(os.path.dirname(__file__), "snapshots"))


atexit.register(clear_snapshots)

if __name__ == "__main__":
    main()
