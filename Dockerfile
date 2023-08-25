# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in Docker
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install the required packages
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8111

# Define environment variable (if any)
# ENV NAME=world

# Run your application when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
