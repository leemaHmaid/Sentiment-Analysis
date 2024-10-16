# Use a python image as the base
FROM python:3.11-slim

# Set the working directory
WORKDIR /code

# Copy the requirements to the working directory
COPY ./requirements.txt /code/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy the app directory contents to the working directory
COPY ./streamlit.py /code/streamlit.py

EXPOSE 8501

CMD ["streamlit","run", "streamlit.py"]