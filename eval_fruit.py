import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging

import research.slim.nets.mobilenet_v1 as mobilenet_v1

from utils.gen_utils import load_batch, get_dataset, load_batch_dense

import os
import time
import datetime

slim = tf.contrib.slim


#=======Dataset Informations=======#

main_dir = "./train_fruit"
log_dir= main_dir + "/log_eval"

#=======Training Informations======#
#Nombre d'époques pour l'entraînement
num_epochs = 1

def evaluate(checkpoint_eval, dataset_dir, file_pattern, file_pattern_for_counting, labels_to_name, batch_size, image_size):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    #Create log_dir:
    with tf.Graph().as_default():
    #=========== Evaluate ===========#
        global_step_cs = tf.train.get_or_create_global_step()
        # Adding the graph:

        dataset = get_dataset("validation", dataset_dir, file_pattern=file_pattern, file_pattern_for_counting=file_pattern_for_counting, labels_to_name=labels_to_name)

        #load_batch_dense is special to densenet or nets that require the same preprocessing
        images,_, oh_labels, labels = load_batch_dense(dataset, batch_size, image_size, image_size, is_training=False)

        #Calcul of batches/epoch, number of steps after decay learning rate
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed


        #Create the model inference
        with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=False)):
            logits, end_points = mobilenet_v1.mobilenet_v1_050(images, num_classes = len(labels_to_name), is_training = False)
        
        """net = slim.dropout(net, keep_prob=0.5, scope='Dropout_1b')
        net = slim.conv2d(net, 512, [1,1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')
        net = slim.dropout(net, keep_prob=0.5, scope='Dropout_1b')
        net = slim.conv2d(net, 256, [1,1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1_1', )
        logits = slim.conv2d(net, dataset.num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1_2')
        logits = tf.nn.relu(logits, name='final_relu')
        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')"""
        variables_to_restore = slim.get_variables_to_restore()
        end_points['Predictions'] = logits
        #Defining accuracy and predictions:
    
        predictions = tf.argmax(end_points['Predictions'], 1)
        labels = tf.squeeze(labels)
        probabilities = end_points['Predictions']

        #Define the metrics to evaluate
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy_validation': slim.metrics.streaming_accuracy(predictions, labels),
        })
        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
        #Define and merge summaries:
        tf.summary.histogram('Predictions_validation', probabilities)
        summary_op_val = tf.summary.merge_all()

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
            summary_op = summary_op_val,
            variables_to_restore = variables_to_restore)
