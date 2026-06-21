# Use an official PyTorch runtime with CUDA for GPU support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

# Expose the API port (Hugging Face Spaces)
ENV PORT=7860
EXPOSE 7860

# Run the API with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
