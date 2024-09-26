# Use an official Python runtime as a parent image
FROM alpine:3.19

# Set the working directory in the container
WORKDIR /POC_CLUSTERING

# Copy the current directory contents into the container at /app
COPY . .

# Install any necessary dependencies
RUN pip install -r requirements.txt

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run your FastAPI app with Uvicorn
CMD ["uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"]
