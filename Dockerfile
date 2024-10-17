FROM nvcr.io/nvidia/pytorch:22.08-py3
# see https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-08.html

# Copy code and install package
COPY . /app  
WORKDIR /app
RUN cd /app && pip install -e .

# Install ollama
RUN curl -fsSL https://ollama.com/install.sh | sh