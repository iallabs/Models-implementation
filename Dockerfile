FROM medtune/tensorflow:python-3.5

RUN apt update && apt install -y git

RUN git clone https://github.com/tensorflow/models

RUN cp -r models/research .

RUN cd research && \
    python3 setup.py install
RUN cd research/slim && \
    python3 setup.py install

ADD . .

#RUN python3 train_mura.py --dataset_dir=MURA-v1.1 --train_dir=logs/train --ckpt=./ckpt_mobilenet/mobilenet_v2_1.4_224.ckpt

ENTRYPOINT [ "python3", "train_mura.py", "--dataset_dir=MURA-v1.1", "--train_dir=logs/train", "--ckpt=./ckpt_mobilenet/mobilenet_v2_1.4_224.ckpt" ]