import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging

import research.slim.nets.mobilenet.mobilenet_v2 as mobilenet_v2

from utils.gen_utils import load_batch, get_dataset, load_batch_dense

import os
import time
import datetime

slim = tf.contrib.slim


#=======Dataset Informations=======#
dataset_dir = "D:/MURA-v1.1/"
main_dir = os.getcwd()
log_dir= main_dir + "\\train\\log_eval"
file_pattern = "MURA_%s_*.tfrecord"
file_pattern_for_counting = "MURA"
batch_size = 8
image_size = 224
labels_to_name= {0:"negative", 1:"positive"}
train_dir = main_dir+"\\train\\training"
#=======Training Informations======#
#Nombre d'époques pour l'entraînement
def evaluate(checkpoint_eval, dataset_dir, file_pattern, file_pattern_for_counting, labels_to_name, batch_size, image_size):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    #Create log_dir:
    with tf.Graph().as_default():
    #=========== Evaluate ===========#
        # Adding the graph:
        tf.logging.set_verbosity(tf.logging.INFO)
        global_step = tf.train.get_or_create_global_step()
        global_step = tf.assign(global_step, global_step+1)
        dataset= get_dataset("validation", dataset_dir, file_pattern=file_pattern,
                             file_pattern_for_counting=file_pattern_for_counting, labels_to_name=labels_to_name)

        #load_batch_dense is special to densenet or nets that require the same preprocessing
        images,_,_, oh_labels, labels = load_batch_dense(dataset, batch_size, image_size, image_size,
                                                         is_training=False, shuffle=False)

        #Calcul of batches/epoch, number of steps after decay learning rate
        num_batches_per_epoch = int(dataset.num_samples / batch_size)


        #Create the model inference
        with slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
            #TODO: Check mobilenet_v1 module, var "excluding
            logits, end_points = mobilenet_v2.mobilenet(images,depth_multiplier=1.4, num_classes = len(labels_to_name))
        end_points['Predictions_1'] = tf.nn.softmax(logits)
        variables_to_restore = slim.get_variables_to_restore()
        
        #Defining accuracy and predictions:
        loss = tf.losses.softmax_cross_entropy(onehot_labels = oh_labels, logits = logits)
        total_loss = tf.reduce_mean(tf.losses.get_total_loss())
        predictions = tf.argmax(end_points['Predictions_1'], 1)
        probabilities = end_points['Predictions_1']

        #Define the metrics to evaluate
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy_validation': tf.metrics.accuracy(tf.argmax(oh_labels,1), predictions),
        })
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        #Define and merge summaries:
        tf.summary.scalar('eval/loss', total_loss)
        tf.summary.histogram('Predictions_validation', probabilities)
        summary_op_val = tf.summary.merge_all()
        #This is the common way to define evaluation using slim

        slim.evaluation.evaluate_once(
            master = '',  
            checkpoint_path = checkpoint_eval,
            logdir = log_dir,
            num_evals = num_batches_per_epoch,
            eval_op = list(names_to_updates.values()),
            variables_to_restore = variables_to_restore,
            summary_op=summary_op_val)

ckpt_eval = tf.train.get_checkpoint_state(train_dir).model_checkpoint_path

evaluate(ckpt_eval, dataset_dir, file_pattern, file_pattern_for_counting, labels_to_name, batch_size, image_size)