# Image for Google Cloud Run deployment
# Use an official Python runtime as a parent image
FROM python:3.10-slim as builder

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
COPY ./app/backend /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache

# Install gcsfuse for Google Cloud Storage access to model
#---------------------------------------------
# Update and install necessary packages
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    lsb-release \
    # Additional dependencies that might be missing in the slim image
    fuse

# Add the gcsfuse distribution URL as a package source
RUN export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` && \
    echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list

# Import the GCSFuse public key
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -

# Install gcsfuse
RUN apt-get update
RUN apt-get install -y gcsfuse 

# Clean up APT when done
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
# End of gcsfuse installation
#---------------------------------------------

# Create a directory for the bucket
RUN mkdir /mnt/fruity-model-registry


# Start a new stage
FROM python:3.10-slim

WORKDIR /app

# Copy only the dependencies installation from the 1st stage image
COPY --from=builder /usr/local /usr/local

# Copy the source code from the 1st stage image
COPY --from=builder /app /app

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD sh -c 'gcsfuse my-gcs-bucket /mnt/my-gcs-bucket && uvicorn your_fastapi_app:app --host 0.0.0.0 --port $PORT'