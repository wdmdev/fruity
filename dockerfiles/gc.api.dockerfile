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

#execute python script setup_models.py
RUN mkdir models
RUN python setup_model.py

# Make port 80 available to the world outside this container
EXPOSE 80

# Connect cloud buckt as folder and run api when the container launches
CMD ["sh", "-c", "uvicorn fruity_api:app --host 0.0.0.0 --port $PORT"]
