## Use Python 3.11 slim image as base
#FROM python:3.11-slim
#
## Set working directory in container
#WORKDIR /app
#
## Copy requirements first to leverage Docker cache
#COPY requirements.txt .
#
## Install system dependencies (Poppler, Tesseract-OCR) and Python packages
#RUN apt-get update && \
#    apt-get install -y --no-install-recommends \
#    build-essential \
#    poppler-utils \
#    tesseract-ocr \
#    tesseract-ocr-eng \
#    && pip install --no-cache-dir -r requirements.txt \
#    && apt-get remove -y build-essential \
#    && apt-get autoremove -y \
#    && apt-get clean \
#    && rm -rf /var/lib/apt/lists/*
#
## Copy the rest of application code
#COPY . .
#
## Expose port
#EXPOSE 8000
#
## Command to run the application with uvicorn
#CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

# Use the official AWS Lambda Python runtime as the base image
FROM public.ecr.aws/lambda/python:3.8

# Set the working directory inside the container
WORKDIR /var/task

# Install your dependencies (e.g., if you have a requirements.txt)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the function code into the container
COPY app.py .

# Command to run your function
CMD ["lambda_function.handler"]
