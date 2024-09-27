# Use the official PyTorch image with CUDA support
FROM pytorch/pytorch:latest

# Install git using apt
RUN apt-get update && apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install necessary dependencies
RUN pip install --upgrade pip
RUN pip install transformers torch
RUN pip install transformers torch sentencepiece tiktoken protobuf accelerate>=0.26.0 datasets wikipedia-api