"""Streamlit frontend for the Fruity app."""
import os
import streamlit as st
from PIL import Image
import io


def main():
    """Start the frontend."""
    st.title("DEMO: fruity classification")

    col1, col2, col3 = st.columns(3)

    with col1:
        picture1 = st.camera_input("Take picture 1")
        if picture1:
            st.image(picture1)
            process_image(picture1, "Image_1")

    with col2:
        picture2 = st.camera_input("Take picture 2")
        if picture2:
            st.image(picture2)
            process_image(picture2, "Image_2")

    with col3:
        picture3 = st.camera_input("Take picture 3")
        if picture3:
            st.image(picture3)
            process_image(picture3, "Image_3")
    # # Capture picture from webcam
    # picture = st.camera_input("Take a picture")

    # # Button to capture a new image
    # if st.button('Capture New Image'):
    #     # This will clear the existing image and prompt for a new one
    #     st.experimental_rerun()

    # # Display the captured picture
    # if picture:
    #     st.image(picture)
    #     process_image(picture)


def process_image(uploaded_file, image_label):
    """Process the image with your ML model and display the results."""
    # Convert the uploaded file to an image
    if uploaded_file is not None:
        # To read image file buffer with PIL:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data))

        # Save the image to a temporary directory
        path = os.path.join(os.path.dirname(__file__), "snapshots", image_label + ".png")
        image.save(path)

        # Here you can add the code to process the image with your ML model
        # For example, predict_label = my_ml_model.predict(image)
        # st.write("Predicted Label: ", predict_label)

        # For demonstration, just displaying the image size
        st.write("Image size: ", image.size)


if __name__ == "__main__":
    main()
