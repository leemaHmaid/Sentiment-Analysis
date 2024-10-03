FROM python:3.10-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# ENV SECRET_KEY=${SECRET_KEY}

EXPOSE 8080

COPY . /code/

CMD ["fastapi", "run", "src/deployment/app.py", "--port", "8080"]