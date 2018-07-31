FROM tensorflow/tensorflow:nightly-gpu-py3 

WORKDIR /medtune

RUN apt update && apt install -y git

# Setup research & slim
RUN git clone https://github.com/tensorflow/models

RUN cp -r models/research .

RUN cd research && \
    python3 setup.py install
RUN cd research/slim && \
    python3 setup.py install

# Add all project files and dirs
ADD . .

RUN mkdir -p logs/train

# Entrypoint will get executed after starting the container
ENTRYPOINT [ "python3", "train_mura.py", "--dataset_dir=MURA-v1.1", "--train_dir=logs/train", "--ckpt=./ckpt_mobilenet/mobilenet_v2_1.4_224.ckpt" ]


# Create container image from this file:
# docker build -t train-mura-image .


# Run container from image :
# nvidia-docker run -dti \                                                                          #Run in detached (-d) and interactive (-i) modes
#  --name=train_mura \                                                                              #Container name
#  -v=/mnt/disks/mdtn-s2/data/models-implementation/data/MURA-v1.1:/medtune/MURA-v1.1 \             #Share MURA data 
#  -v=/mnt/disks/mdtn-s2/data/models-implementation/ckpt/ckpt_mobilenet:/medtune/ckpt_mobilenet \   #Share CKPT
#  -v=./logs/train:/medtune/logs/train \                                                            #Share training logs
#  train-mura-image                                                                                 #Image name