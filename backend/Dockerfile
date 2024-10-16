# Use a python image as the base
FROM python:3.11-slim

# Set the working directory
WORKDIR /code

# Copy the requirements to the working directory
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the app directory contents to the working directory
COPY ./app /code/app

# Copy the models into working directory
COPY ./models /code/models
COPY ./configs /code/configs
COPY ./db /code/db
COPY ./schemas /code/schemas
COPY ./utils /code/utils

EXPOSE 8080

# Run the FastAPI application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080" ,"--reload"]