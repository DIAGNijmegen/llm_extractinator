FROM nvidia/cuda:12.6.2-devel-ubuntu24.04

# Install Python and other dependencies
RUN apt-get update && apt-get install -y \
    python3.13 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set default Python version
RUN ln -sf /usr/bin/python3.13 /usr/bin/python

# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy code and install package
COPY . /app  
WORKDIR /app
RUN cd /app && pip install -e .
RUN pip install ollama --upgrade