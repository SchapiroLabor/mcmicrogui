FROM python:3.9

RUN apt-get update && apt-get install -y python3-opencv

COPY . mcmicrogui

RUN pip3 install -r /mcmicrogui/requirements.txt
