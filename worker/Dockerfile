FROM python:3.8-buster

RUN apt-get update
RUN apt-get install -y openjdk-11-jdk-headless

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# create mar
RUN mkdir -p model_store
RUN torch-model-archiver --model-name mnist_model --version 1.0 --serialized-file best_chk.pt --model-file model.py --handler handler.py -f
RUN mv mnist_model.mar model_store

# start torchserve
CMD torchserve --start --ncs --model-store model_store --models mymodel=mnist_model.mar --foreground --ts-config config.properties
