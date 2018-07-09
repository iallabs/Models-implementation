import tensorflow as tf
slim = tf.contrib.slim
import research.slim.nets.mobilenet_v1 as mobilenet_v1
import research.slim.datasets.imagenet as imagenet
from tensorflow.python.platform import gfile
from research.slim.preprocessing import inception_preprocessing
import DenseNet.preprocessing.densenet_pre as dp
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np

checkpoint_dir = os.getcwd()
checkpoint_file = checkpoint_dir + "/train_fruit/training/model.ckpt-2316"

image_size = 224
main_dir = "D:/train/train/"
#Images
labels_to_name = {0:'Apple Braeburn', 
                1:'Apple Golden 1',
                2:'Apple Golden 2', 
                3:'Apple Golden 3',
                4: 'Apple Granny Smith',
                5: 'Apple Red 1',
                6: 'Apple Red 2',
                7: 'Apple Red 3',
                8: 'Apple Red Delicious',
                9: 'Apple Red Yellow',
                10: 'Apricot',
                11: 'Avocado',
                12: 'Avocado ripe',
                13: 'Banana',
                14: 'Banana Red',
                15: 'Cactus fruit',
                16: 'Cantaloupe 1',
                17: 'Cantaloupe 2',
                18: 'Carambula',
                19: 'Cherry 1',
                20: 'Cherry 2',
                21: 'Cherry Rainier',
                22: 'Clementine',
                23: 'Cocos',
                24: 'Dates',
                25: 'Granadilla',
                26: 'Grape Pink',
                27: 'Grape White',
                28: 'Grape White 2',
                29: 'Grapefruit Pink',
                30: 'Grapefruit White',
                31: 'Guava',
                32: 'Huckleberry',
                33: 'Kaki',
                34: 'Kiwi',
                35: 'Kumquats',
                36: 'Lemon',
                37: 'Lemon Meyer',
                38: 'Limes',
                39: 'Litchi',
                40: 'Mandarine',
                41: 'Mango',
                42: 'Maracuja',
                43: 'Melon Piel de Sapo',
                44: 'Nectarine',
                45: 'Orange',
                46: 'Papaya',
                47: 'Passion Fruit',
                48: 'Peach',
                49: 'Peach Flat',
                50: 'Pear',
                51: 'Pear Abate',
                52: 'Pear Monster',
                53: 'Pear Williams',
                54 : 'Pepino',
                55: 'Pineapple',
                56: 'Pitahaya Red',
                57: 'Plum',
                58: 'Pomegranate',
                59: 'Quince',
                60: 'Raspberry',
                61: 'Salak',
                62: 'Strawberry',
                63: 'Tamarillo',
                64: 'Tangelo'
                }

tf.logging.set_verbosity(tf.logging.INFO)


sample_images = ["D:/ChestXray-14/images/00000001_000.png","D:/fruits/fruits-360/Training/"+labels_to_name[0]+"/0_100.jpg",
                  "C:/Users/Lenovo/Documents/testset/kaki-2.jpg"]
file_input = tf.placeholder(tf.string, ())
image = tf.image.decode_jpeg(tf.read_file(file_input), channels=3)

image_a = inception_preprocessing.preprocess_image(image, 224,224, is_training=False)

"""images = dp.preprocess_image(image, 224, 224, is_training=False)"""
images_bis = tf.expand_dims(image_a,0)
with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=False)):
  logits, endpoints = mobilenet_v1.mobilenet_v1_050(images_bis, num_classes=len(labels_to_name), is_training=False)

endpoints['Predictions'] = tf.nn.sigmoid(logits)


with tf.Session() as sess:
  vars = slim.get_variables_to_restore()
  saver = tf.train.Saver(vars)
  saver.restore(sess,  checkpoint_file)
  
  print(image_a.eval(feed_dict={file_input: sample_images[2]}))

  x = endpoints['Predictions'].eval(feed_dict={file_input: sample_images[2]})

  print("Prediction class:", labels_to_name[x.argmax()])
  print("Prediction value", x.max())
