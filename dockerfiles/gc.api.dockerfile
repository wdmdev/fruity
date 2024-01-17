# Image for Google Cloud Run deployment
# Use an official Python runtime as a parent image
FROM python:3.10-slim as builder

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
COPY ./app/backend /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache

# Start a new stage
#---------------------------------------------
FROM python:3.10-slim

WORKDIR /app

# Copy only the dependencies installation from the 1st stage image
COPY --from=builder /usr/local /usr/local

# Copy the source code from the 1st stage image
COPY --from=builder /app /app

# Install gsutil
#---------------------------------------------
# Install curl and other dependencies
RUN apt-get update && apt-get install -y curl gnupg

# Add the Cloud SDK distribution URI as a package source
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list

# Import the Google Cloud public key
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg add -

# Update the package list and install the Cloud SDK
RUN apt-get update && apt-get install -y google-cloud-sdk
#---------------------------------------------

# Load model from cloud bucket
#---------------------------------------------
RUN mkdir models

# Set the bucket and blob name
ENV BUCKET_NAME="fruity-model-registry"
ENV BLOB_NAME="model.pth"

# Set the destination file name
ENV DESTINATION_FILE_NAME="models/model.pth"

# Download the blob
RUN gsutil cp gs://$BUCKET_NAME/$BLOB_NAME $DESTINATION_FILE_NAME
#---------------------------------------------

# Make port 80 available to the world outside this container
EXPOSE 80

# Connect cloud buckt as folder and run api when the container launches
CMD ["sh", "-c", "uvicorn fruity_api:app --host 0.0.0.0 --port $PORT"]
