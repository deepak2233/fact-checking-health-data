# Dockerfile
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the source code into the container
COPY src src

# Copy the models directory into the container
COPY models models

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]


