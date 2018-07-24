import tensorflow as tf
slim = tf.contrib.slim

import research.slim.nets.mobilenet.mobilenet_v2 as mobilenet_v2
import research.slim.datasets.imagenet as imagenet
from tensorflow.python.platform import gfile
from research.slim.preprocessing import inception_preprocessing
import DenseNet.preprocessing.densenet_pre as dp
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

checkpoint_dir = os.getcwd()
checkpoint_file = os.getcwd()+"\\train\\training\\model-16100"

image_size = 224
#Images
labels_to_name = {0:'negative', 
                1:'positive'
                }

tf.logging.set_verbosity(tf.logging.INFO)


sample_images = ["D:/MURA-v1.1/MURA-v1.1/valid/XR_SHOULDER/patient11259/study1_negative/image1.png"]
file_input = tf.placeholder(tf.string, ())
image = tf.image.decode_png(tf.read_file(file_input), channels=3)
image = tf.image.convert_image_dtype(image, tf.float32)
image.set_shape([None,None,3])
image_a = dp.preprocess_image(image, 224,224, is_training=False)
images_bis = tf.expand_dims(image_a,0)
with slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
  logits, endpoints = mobilenet_v2.mobilenet(images_bis,depth_multiplier=1.4, num_classes = len(labels_to_name))

endpoints['Predictions'] = tf.nn.softmax(logits)
txt_file = open("Inference.txt", "w")
with tf.Session() as sess:
  variables = slim.get_variables_to_restore()

  saver = tf.train.Saver(variables)
  saver.restore(sess,  checkpoint_file)
  a = images_bis.eval(feed_dict={file_input: sample_images[0]})
  l = logits.eval(feed_dict={file_input: sample_images[0]})
  for k in a:
    for i in k:
      txt_file.write(str(i))
  y = logits.eval(feed_dict={file_input: sample_images[0]})
  print(y)
  x = endpoints['Predictions'].eval(feed_dict={file_input: sample_images[0]})
  print("Prediction class:", labels_to_name[x.argmax()])
  print("Prediction value", x.max())
txt_file.close()