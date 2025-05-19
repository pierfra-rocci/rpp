# Use official Python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


COPY . /app
# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y gcc build-essential libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/*

# install
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# This command creates a .streamlit directory in the home directory of the container.
RUN mkdir ~/.streamlit

# This copies your Streamlit configuration file into the .streamlit directory you just created.
RUN cp .streamlit/config.toml ~/.streamlit/config.toml

# Similar to the previous step, this copies your Streamlit credentials file into the .streamlit directory.
RUN cp .streamlit/credentials.toml ~/.streamlit/credentials.toml

# Expose ports for Flask (5000) and Streamlit (8501)
EXPOSE 5000 8501

# Create a script to run both backend and frontend
RUN echo '#!/bin/bash\n\
python backend_pro.py &\n\
streamlit run frontend.py --server.port 80 --server.address 0.0.0.0\n' > /app/start.sh && chmod +x /app/start.sh

# Default command
CMD ["/bin/bash", "/app/start.sh"]
