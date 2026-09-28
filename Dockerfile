# Start from a small Linux system with the same Python as your Mac (3.13)
FROM python:3.13-slim

# LightGBM needs this system library to run on Linux
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Work inside a folder called /app
WORKDIR /app

# Install the libraries first (Docker caches this step, so rebuilds are fast)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy in the app code and the trained model
COPY api.py app.py ./
COPY model/ model/