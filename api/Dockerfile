FROM python:3.8-buster

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD ["gunicorn", "--timeout", "1000", "--workers", "2", "--bind", "0.0.0.0:5001", "main:app"]