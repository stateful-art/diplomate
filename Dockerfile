# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all Python scripts into the container
COPY *.py /app/

# Copy the output folder (with trained models or data) into the container
COPY output /app/output

# Make sure all Python scripts are executable
RUN chmod +x /app/*.py
