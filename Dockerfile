FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -r requirements-test.txt
RUN pip install -r requirements-dev.txt

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000"]