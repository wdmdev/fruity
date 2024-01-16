# Use an official Python runtime as a parent image
FROM python:3.10-slim as builder

# Set the working directory in the container to /app
WORKDIR /app

#print current directory
RUN pwd
#print contents of current directory
RUN ls -la

# Add the current directory contents into the container at /app
COPY ./app/backend /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && rm -rf /root/.cache

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
CMD ["uvicorn", "fruity_api:app", "--host", "0.0.0.0", "--port", "80"]