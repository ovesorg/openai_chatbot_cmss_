# Use an official Python runtime as the base image
# FROM python:3.9
FROM alpine:latest
#FROM python:3.9-alpine

#install python on top of alpine
RUN apk --no-cache add python3 py3-pip

# Set the working directory in Docker
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install necessary tools and the required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the OpenAI API key as an environment variable
ARG OPENAI_API_KEY
ENV OPENAI_API_KEY=$OPENAI_API_KEY

# Set the Pinecone API key as an environment variable
ARG PINECONE_API_KEY
ENV PINECONE_API_KEY=$PINECONE_API_KEY

# Make port 80 available to the world outside this container
EXPOSE 8111

# Define environment variable (if any)
# ENV NAME=world

# Run your application when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8111"]
