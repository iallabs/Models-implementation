import tensorflow as tf
slim = tf.contrib.slim
from utils.utils_csv import _get_infos
import DenseNet.nets.densenet as densenet
from research.slim.preprocessing import inception_preprocessing
import DenseNet.preprocessing.densenet_pre as dp

import pandas as pd
import os
import numpy as np
dataset_dir="D:/ChestXray-14/images/"
checkpoint_dir = os.getcwd()
checkpoint_file = os.getcwd()+"\\train\\training\\model-50448"

image_size = 224
#Images
labels_to_name = {
                0:'No Finding', 
                1:'Atelectasis',
                2:'Cardiomegaly', 
                3:'Effusion',
                4: 'Infiltration',
                5: 'Mass',
                6: 'Nodule',
                7: 'Pneumonia',
                8: 'Pneumothorax',
                9: 'Consolidation',
                10: 'Edema',
                11: 'Emphysema',
                12: 'Fibrosis',
                13: 'Pleural_Thickening',
                14: 'Hernia'
                }

tf.logging.set_verbosity(tf.logging.INFO)
grouped = _get_infos("D:/ChestXray-14/","valid_chest_paths.csv")

file_input = tf.placeholder(tf.string, ())
image = tf.image.decode_png(tf.read_file(file_input), channels=3)
image = tf.image.convert_image_dtype(image, tf.float32)
image.set_shape([None,None,3])
image_a = dp.preprocess_image(image, 224,224, is_training=False)
images_bis = tf.expand_dims(image_a,0)
with slim.arg_scope(densenet.densenet_arg_scope(is_training=False)):
  logits, endpoints = densenet.densenet121(images_bis, num_classes = len(labels_to_name), is_training = False)
variables = slim.get_variables_to_restore()
saver = tf.train.Saver(variables)
endpoints['Predictions'] = tf.nn.softmax(logits)
txt_file = open("Inference-chest-50448.txt", "w")
with tf.Session() as sess:
  saver.restore(sess,  checkpoint_file)
  for i in range(len(grouped)):
    row = grouped.iloc[i]
    y = endpoints['Predictions'].eval(feed_dict={file_input: dataset_dir+row[0]})
    txt_file.write("image %d,prediction class %d ,prediction value %.4f, image path:%s \n"%(i,y.argmax(),y.max(),row[0]))
txt_file.close()