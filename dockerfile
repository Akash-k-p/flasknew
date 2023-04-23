FROM python:3.8-slim-buster

RUN apt-get update && \
    apt-get -qq -y install tesseract-ocr && \
    apt-get -qq -y install libtesseract-dev

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata/

CMD ["gunicorn", "app:app","--timeout","120"]