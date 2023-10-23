# Use an official Python runtime as a parent image
FROM python:3.10-slim-buster

# Set the working directory in docker
WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

COPY app.py /app

# Make port 3031 available to the world outside this container
EXPOSE 3031

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
