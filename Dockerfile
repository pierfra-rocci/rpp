# Use official Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc build-essential libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements (if exists) and install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose ports for Flask (5000) and Streamlit (8501)
EXPOSE 5000 8501

# Create a script to run both backend and frontend
RUN echo '#!/bin/bash\n\
python backend_dev.py &\n\
streamlit run frontend.py --server.port 8501 --server.address 0.0.0.0\n' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/bin/bash", "/app/start.sh"]
