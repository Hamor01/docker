# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any dependencies needed for your Python application
RUN pip install --no-cache-dir tkinter

# Command to run your Python application
CMD ["python3", "healthcare.py"]