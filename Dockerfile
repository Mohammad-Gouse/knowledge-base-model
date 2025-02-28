# Use a Python base image
FROM python:3.10-slim

# Set environment variables (use quotes for special characters)
ENV ACCESS_TOKEN="EAAWA3MqEIZBMBO2jAXsVW1fWrBq6o0NZBjybzl6nGYnwl9VkEbRa3MWIQgQjFqZBwJRtkdUOt0Bq2V4ADQzT1RotjM24xTNiAzlartIlH2ftPkKqNf57b3oyfl5aBhRiGhzZBNbiBCIhOKZAZAbJXz6K0ao9D3rLWPvKgvIkZBHKvKvsSWESKK3z5hDFUdMJPnZAUa09EswpBqNKvZBcuzo9QklRUXQZB2DriZBf5myRUGx"
ENV VERIFY_TOKEN="my_verify_token"
ENV AWS_REGION="ap-south-1"

# Numeric values (optional to wrap in quotes)
ENV MAX_TOKENS="500"
ENV TEMPERATURE="0.5"
ENV TOP_P="0.9"
ENV TOP_K="50"
ENV THRESHOLD="1.8"
ENV CHUNKS="3"

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements.txt to the working directory
COPY requirements.txt /app/

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-eng -y

# Install dependencies directly in the container
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . /app/

# Expose the port the app will run on
EXPOSE 8000

# Run the FastAPI app using uvicorn directly
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
