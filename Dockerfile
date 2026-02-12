# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /code

# Copy the requirements file into the container at /code
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a directory for the database and give permission to write
# (This fixes the "Database is locked/read-only" error on Cloud)
RUN chmod -R 777 /code

# Make sure the app listens on port 7860 (Hugging Face requirement)
ENV PORT=7860

# Run the app using Gunicorn server
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]