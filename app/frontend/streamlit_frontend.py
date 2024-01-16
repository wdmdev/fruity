"""Streamlit frontend for the Fruity app."""
import streamlit as st
from PIL import Image
import io

def main():
    """Start the frontend."""
    st.title("Webcam Picture Capture with ML Classification")

    # Capture picture from webcam
    picture = st.camera_input("Take a picture")

    # Button to capture a new image
    if st.button('Capture New Image'):
        # This will clear the existing image and prompt for a new one
        st.experimental_rerun()

    # Display the captured picture
    if picture:
        st.image(picture)
        process_image(picture)


def process_image(uploaded_file):
    """Process the image with your ML model and display the results."""
    # Convert the uploaded file to an image
    if uploaded_file is not None:
        # To read image file buffer with PIL:
        bytes_data = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(bytes_data))

        # Here you can add the code to process the image with your ML model
        # For example, predict_label = my_ml_model.predict(image)
        # st.write("Predicted Label: ", predict_label)

        # For demonstration, just displaying the image size
        st.write("Image size: ", image.size)

if __name__ == "__main__":
    main()
