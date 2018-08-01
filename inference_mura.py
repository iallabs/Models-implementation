import tensorflow as tf
slim = tf.contrib.slim
from utils.utils_csv import _get_infos
import research.slim.nets.mobilenet.mobilenet_v2 as mobilenet_v2
import research.slim.datasets.imagenet as imagenet
from tensorflow.python.platform import gfile
from research.slim.preprocessing import inception_preprocessing
import DenseNet.preprocessing.densenet_pre as dp

import pandas as pd
import os
import numpy as np
dataset_dir="D:/MURA-v1.1/"
checkpoint_dir = os.getcwd()
checkpoint_file = os.getcwd()+"\\train\\training\\model-25300"

image_size = 224
#Images
labels_to_name = {0:'negative', 
                1:'positive'
                }

tf.logging.set_verbosity(tf.logging.INFO)
grouped = _get_infos("D:/MURA-v1.1","valid_image_paths.csv")

file_input = tf.placeholder(tf.string, ())
image = tf.image.decode_image(tf.read_file(file_input), channels=3)
image = tf.image.convert_image_dtype(image, tf.float32)
image.set_shape([None,None,3])
image_a = dp.preprocess_image(image, 224,224, is_training=False)
images_bis = tf.expand_dims(image_a,0)
logits, endpoints = mobilenet_v2.mobilenet(images_bis,depth_multiplier=1.4, num_classes = len(labels_to_name))
init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, slim.get_model_variables('MobilenetV2'))

endpoints['Predictions'] = tf.nn.softmax(logits)
totalacc = 0.
txt_file = open("Inference-mura-25300.txt", "w")
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(len(grouped)):
    row = grouped.iloc[i]
    class_name = row[0].split('/')[-2].split('_')[-1]
    y, logit = sess.run([endpoints['Predictions'], logits], feed_dict={file_input: dataset_dir+row[0]})

    txt_file.write("image %d, prediction class %s, prediction value %.4f, image path:%s \n"%(i,labels_to_name[y.argmax()],y.max(),row[0]))
txt_file.close()