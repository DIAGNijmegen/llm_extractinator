FROM nvcr.io/nvidia/pytorch:22.08-py3
# see https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel_22-08.html

# Install dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . /app
WORKDIR /app