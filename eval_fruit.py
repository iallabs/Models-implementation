import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging

import research.slim.nets.mobilenet_v1 as mobilenet_v1

from utils.gen_utils import load_batch, get_dataset

import os
import time
import datetime

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('dataset_dir',None,'String: Your dataset directory')
flags.DEFINE_string('ckpt_number','0','String: ckpt file number')
FLAGS = flags.FLAGS
#=======Dataset Informations=======#
dataset_dir = FLAGS.dataset_dir
ckpt_num = FLAGS.ckpt_number


main_dir = "./train_fruit"
log_dir= main_dir + "/log_eval"


#Emplacement du checkpoint file
checkpoint_dir = main_dir +"/training/model.ckpt-"+ ckpt_num
image_size = 224
#Nombre de classes à prédire
num_class = 65

file_pattern = "fruit_%s_*.tfrecord"
file_pattern_for_counting = "fruit"
#Création d'un dictionnaire pour reférer à chaque label
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
#Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.

#=======Training Informations======#
#Nombre d'époques pour l'entraîen
num_epochs = 1

#State your batch size
batch_size = 16

def run():
    #Create log_dir:
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

    with tf.Graph().as_default():
    #=========== Evaluate ===========#
        global_step_cs = tf.train.get_or_create_global_step()
        # Adding the graph:

        dataset = get_dataset("validation", dataset_dir, file_pattern=file_pattern, file_pattern_for_counting=file_pattern_for_counting, labels_to_name=labels_to_name)
        images,_, oh_labels, labels = load_batch(dataset, batch_size, image_size, image_size, is_training=False)

        #Calcul of batches/epoch, number of steps after decay learning rate
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed


        #Create the model inference
        """with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):"""
        logits, end_points = mobilenet_v1.mobilenet_v1_050(images, num_classes = dataset.num_classes, is_training = False)
        variables_to_restore = slim.get_variables_to_restore()
       
        #Defining accuracy and predictions:
    
        predictions = tf.argmax(end_points['Predictions'], 1)
        labels = tf.squeeze(labels)
        probabilities = end_points['Predictions']

        #Define the metrics to evaluate
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        })
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        #Define and merge summaries:
        tf.summary.histogram('Predictions', probabilities)
        summary_op = tf.summary.merge_all()

        #This is the common way to define evaluation using slim
        max_step = num_epochs*num_steps_per_epoch
        initial_op=tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        slim.evaluation.evaluate_once(
            master = '',  
            checkpoint_path = checkpoint_dir,
            logdir = log_dir,
            num_evals = max_step,
            initial_op = initial_op,
            eval_op = list(names_to_updates.values()),
            summary_op = summary_op,
            variables_to_restore = variables_to_restore)

if __name__ == '__main__':
    run()