import tensorflow as tf


from tensorflow.python.platform import tf_logging as logging

import research.slim.nets.mobilenet_v1 as mobilenet_v1

from utils.gen_utils import load_batch, get_dataset, load_batch_dense
from eval_fruit import evaluate

import os
import sys
import time
import datetime

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_float('gpu_p', 1.0, 'Float: allow gpu growth value to pass in config proto')
flags.DEFINE_string('dataset_dir','D:/fruits/fruits-360','String: Your dataset directory')
flags.DEFINE_string('train_dir', 'train_fruit/training', 'String: Your train directory')
flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
flags.DEFINE_string('ckpt','train_fruit/net/mobilenet_v1_0.5_160.ckpt','String: Your dataset directory')
FLAGS = flags.FLAGS

#=======Dataset Informations=======#
dataset_dir = FLAGS.dataset_dir
gpu_p = FLAGS.gpu_p
train_dir = FLAGS.train_dir

#Emplacement du checkpoint file
checkpoint_file= FLAGS.ckpt

image_size = 224
#Nombre de classes à prédire
file_pattern = "traincvsd_%s_*.tfrecord"
file_pattern_for_counting = "traincvsd"
"""file_pattern = "fruit_%s_*.tfrecord"
file_pattern_for_counting = "fruit"""
#Création d'un dictionnaire pour reférer à chaque label
"""labels_to_name = {0:'Apple Braeburn', 
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
                }"""

"""labels_to_name = {
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
                }"""
labels_to_name = {
    'cat':0,
    'dog':1
}
#=======Training Informations======#
#Nombre d'époques pour l'entraînement
num_epochs = 2


#State your batch size
batch_size = 16

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0001
learning_rate_decay_factor = 0.95
num_epochs_before_decay = 1

def run():
    #Create log_dir:
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)

    #===================================================================== Training ===========================================================================#
    #Adding the graph:
    tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level:
    with tf.name_scope("dataset"):
        dataset = get_dataset("train", dataset_dir, file_pattern=file_pattern, file_pattern_for_counting=file_pattern_for_counting, labels_to_name=labels_to_name)
    with tf.name_scope("load_data"):
        images,_, oh_labels, labels = load_batch(dataset, batch_size, image_size, image_size, shuffle=False, is_training=True)

        #Calcul of batches/epoch, number of steps after decay learning rate
    num_batches_per_epoch = int(dataset.num_samples / batch_size)
    num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
    decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)
    
    #Create the model inference
    with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True, regularize_depthwise=True, weight_decay=0.0004)):
    #TODO: Check mobilenet_v1 module, var "excluding
        logits, end_points = mobilenet_v1.mobilenet_v1_050(images, num_classes = len(labels_to_name), is_training = True)
            
    excluding = ['MobilenetV1/Logits/Conv2d_1c_1x1','MobilenetV1/Logits/Predictions']   
    variable_to_restore = slim.get_variables_to_restore(exclude=excluding)
    saver =tf.train.Saver(variable_to_restore)
    summary = tf.summary.merge_all()
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        print("aaaaaa")
        summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
        print("hello")
        merge , _ = sess.run(summary)
        print("hhhhhh")
        summary_writer.add_summary(merge)
        summary_writer.flush()   

run()