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
RUN apt-get update && apt-get install -y gcsfuse
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