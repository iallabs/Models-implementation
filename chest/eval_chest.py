import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
import DenseNet.nets.densenet as densenet
import DenseNet.preprocessing.densenet_pre as dp
from research.slim.preprocessing import inception_preprocessing
from utils.gen_utils import load_batch, get_dataset, load_batch_dense

import os
import time
import datetime

slim = tf.contrib.slim


#=======Dataset Informations=======#

main_dir = "./train_chest"
log_dir= main_dir + "/eval_chest"


#Emplacement du checkpoint file

image_size = 224
#Nombre de classes à prédire
num_class = 15

file_pattern = "chest%s_*.tfrecord"
file_pattern_for_counting = "chest"
#Création d'un dictionnaire pour reférer à chaque label
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


items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either... ....',
    'label': 'A label that is as such -- fruits'
}

#=======Training Informations======#
#Nombre d'époques pour l'entraîen
num_epochs = 1

#State your batch size
batch_size = 16




def evaluate(checkpoint_eval, dataset_dir, file_pattern, file_pattern_for_counting, labels_to_name, batch_size, image_size):
    #Create log_dir:
    if not os.path.exists(log_dir):
        os.mkdir(os.getcwd+'/'+log_dir)
    tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

    with tf.Graph().as_default():
    #=========== Evaluate ===========#
        global_step_cs = tf.train.get_or_create_global_step()
        # Adding the graph:

        dataset = get_dataset("validation", dataset_dir, file_pattern=file_pattern, file_pattern_for_counting=file_pattern_for_counting, labels_to_name=labels_to_name)
        images,_,_, oh_labels, labels = load_batch_dense(dataset, batch_size, image_size, image_size, is_training=False)

        #Calcul of batches/epoch, number of steps after decay learning rate
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed


        #Create the model inference
        with slim.arg_scope(densenet.densenet_arg_scope(is_training=False)):
            logits, end_points = densenet.densenet121(images, num_classes = dataset.num_classes, is_training = False)
        variables_to_restore = slim.get_variables_to_restore()
        
        
        logit = tf.squeeze(logits)
        predictions = tf.argmax(tf.squeeze(end_points['Predictions']),1)
        
        #Define the metrics to evaluate
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),

        })

        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # merge summaries:

        summary_op = tf.summary.merge_all()

        #This is the common way to define evaluation using slim
        max_step = num_epochs*num_steps_per_epoch
        initial_op=tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        slim.evaluation.evaluate_once(
            master = '',  
            checkpoint_path = checkpoint_eval,
            logdir = log_dir,
            num_evals = max_step,
            initial_op = initial_op,
            eval_op = list(names_to_updates.values()),
            summary_op = summary_op,
            variables_to_restore = variables_to_restore)