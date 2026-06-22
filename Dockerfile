# # Use an official PyTorch runtime with CUDA for GPU support
# FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# # Set working directory inside the container
# WORKDIR /app

# # Install system dependencies required for OpenCV
# RUN apt-get update && apt-get install -y \
#     libgl1-mesa-glx \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

# # Copy requirements and install
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy your source code
# COPY . .

# # Expose the API port (Hugging Face Spaces)
# ENV PORT=7860
# ENV DEEPFACE_HOME=/app
# EXPOSE 7860

# # Run the API with uvicorn
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]


# Use an official PyTorch runtime with CUDA for GPU support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create the persistent storage directory BEFORE copying your app files
RUN mkdir -p /data/evidence

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

# Set up user permissions for Hugging Face persistent volumes (UID 1000)
RUN useradd -m -u 1000 user && \
    chown -R user:user /app /data

# Switch to the non-root user expected by HF Spaces
USER user

# Expose the API port (Hugging Face Spaces)
ENV PORT=7860
ENV DEEPFACE_HOME=/app
EXPOSE 7860

# Run the API with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]