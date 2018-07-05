import tensorflow as tf
slim = tf.contrib.slim
import research.slim.nets.mobilenet_v1 as mobilenet_v1
import research.slim.datasets.imagenet as imagenet
from tensorflow.python.platform import gfile
from research.slim.preprocessing import inception_preprocessing
import DenseNet.preprocessing.densenet_pre as dp
import cv2
import os
import numpy as np

checkpoint_dir = os.getcwd()
checkpoint_file = checkpoint_dir + "/train_fruit/training/model.ckpt-31687"

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


sample_images = ["D:/rasp_2.jpg","D:clemetine_2.jpg"]
file_input = tf.placeholder(tf.string, ())
image = tf.image.decode_jpeg(tf.read_file(file_input), channels=3)


image = dp.preprocess_image(image, 224, 224, is_training=False)
images_bis = tf.expand_dims(image,0)

net, endpoints = mobilenet_v1.mobilenet_v1_050(images_bis, num_classes=None, is_training=False)
net = slim.dropout(net, keep_prob=0.5, scope='Dropout_1b')
net = slim.conv2d(net, 512, [1,1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')
net = slim.dropout(net, keep_prob=0.5, scope='Dropout_1b')
net = slim.conv2d(net, 256, [1,1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1_1')
logits = slim.conv2d(net, len(labels_to_name), [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1_2')
logits = tf.nn.relu(logits, name='final_relu')
logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
logits = tf.nn.softmax(logits)
endpoints['Predictions'] = logits


with tf.Session() as sess:
  vars = slim.get_variables_to_restore()
  saver = tf.train.Saver(vars)
  saver.restore(sess,  checkpoint_file)
  
  x = endpoints['Predictions'].eval(feed_dict={file_input: sample_images[1]})

print("Prediction class:", labels_to_name[x.argmax()])
print("Prediction value", x.max())