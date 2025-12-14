FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
 git \
 libgl1-mesa-glx \
 libglib2.0-0 \
 wget \
 unzip \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./src .
COPY ./notebook ./notebook

RUN chmod +x run.sh

CMD ["bash", "run.sh"]