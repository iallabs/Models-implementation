import tensorflow as tf
slim = tf.contrib.slim
from utils.utils_csv import _get_infos
import research.slim.nets.mobilenet.mobilenet_v2 as mobilenet_v2
from research.slim.preprocessing import inception_preprocessing
import DenseNet.preprocessing.densenet_pre as dp

import pandas as pd
import os
import numpy as np
dataset_dir="D:/MURA-v1.1/"
checkpoint_dir = os.path.join(os.getcwd(),os.path.join("train","training"))
checkpoint_file = os.path.join(checkpoint_dir,"model-56350")

image_size = 224
#Labels 
labels_to_name = {0:'negative', 
                1:'positive'
                }

tf.logging.set_verbosity(tf.logging.INFO)
file_input = tf.placeholder(tf.string, ())
image = tf.image.decode_image(tf.read_file(file_input), channels=3)
image = tf.image.convert_image_dtype(image, tf.float32)
image.set_shape([None,None,3])
image_a = dp.preprocess_image(image, 224,224, is_training=False)
images_bis = tf.expand_dims(image_a,0)
with slim.arg_scope(mobilenet_v2.training_scope(is_training=True)):
    #TODO: Check mobilenet_v1 module, var "excluding
    logits, end_points = mobilenet_v2.mobilenet(images_bis,depth_multiplier=1.4, num_classes = len(labels_to_name))

embedding = end_points["layer_18/output"]

print(end_points)