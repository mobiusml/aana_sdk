# Use NVIDIA CUDA as base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive

# Install required libraries, tools, and Python3
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 curl git python3.10 python3.10-dev python3-pip python3.10-venv

# Install poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Update PATH
RUN echo 'export PATH="/root/.local/bin:$PATH"' >> /root/.bashrc
ENV PATH="/root/.local/bin:$PATH"

# Copy project files into the container
COPY . /app

# Install the package with poetry
RUN sh install.sh

# Prepare the startup script
RUN chmod +x startup.sh

# Disable buffering for stdout and stderr to get the logs in real time
ENV PYTHONUNBUFFERED=1

# Expose the desired port
EXPOSE 8000

# Run the startup script, TARGET can be set with environment variables
ENV TARGET=llama2
CMD ["/app/startup.sh"]
