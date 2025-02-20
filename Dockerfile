FROM python:3.10.9

WORKDIR /app

RUN apt-get update && apt-get install -y \
    pkg-config \
    libhdf5-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    postgresql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt

COPY .deepface/weights /root/.deepface/weights

RUN pip install -r /app/requirements.txt

COPY ./FaceRecognition.py /app/

COPY ./server.py /app/

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]