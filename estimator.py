import tensorflow as tf


from tensorflow.python.platform import tf_logging as logging
import DenseNet.nets.densenet as densenet
import research.slim.nets.mobilenet.mobilenet_v2 as mobilenet_v2
import research.slim.nets.inception_resnet_v2 as inception
from utils.gen_tfrec import load_batch, get_dataset, load_batch_dense

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
train_dir = FLAGS.train_dir
summary_dir = os.path.join(train_dir , "summary")

gpu_p = FLAGS.gpu_p
#Emplacement du checkpoint file
checkpoint_file= FLAGS.ckpt

image_size = 224
#Nombre de classes à prédire
file_pattern = "MURA_%s_*.tfrecord"
file_pattern_for_counting = "MURA"

#Création d'un dictionnaire pour reférer à chaque label

labels_to_name = {
    'negative':0,
    'positive':1
}
#=======Training Informations======#
#Nombre d'époques pour l'entraînement
num_epochs = 100
#State your batch size
batch_size = 32
#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 1e-4
learning_rate_decay_factor = 0.95
num_epochs_before_decay = 1

def run():
    #Create log_dir:
    if not os.path.exists(train_dir):
        os.mkdir(os.path.join(os.getcwd(),train_dir))
    if not os.path.exists(summary_dir):
        os.mkdir(os.path.join(os.getcwd(),summary_dir))
    #===================================================================== Training ===========================================================================#
    #Adding the graph:
    #Set the verbosity to INFO level
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default() as graph:
        with tf.name_scope("dataset"):
            dataset, num_samples= get_dataset("train", dataset_dir, file_pattern=file_pattern,
                                    file_pattern_for_counting=file_pattern_for_counting, labels_to_name=labels_to_name)
        with tf.name_scope("load_data"):
            images, oh_labels = load_batch_dense(dataset, batch_size, image_size, image_size, num_epochs,
                                                            shuffle=True, is_training=True)

        #Calcul of batches/epoch, number of steps after decay learning rate
        num_batches_per_epoch = int(num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        #Create the model inference
        with slim.arg_scope(inception.inception_resnet_v2_arg_scope(weight_decay=5e-4,batch_norm_decay=0.97)):
            #TODO: Check mobilenet_v1 module, var "excluding
            logits, _ = inception.inception_resnet_v2(images, num_classes = len(labels_to_name),create_aux_logits=False, is_training=True)
            
        excluding = ['InceptionResnetV2/Logits/Logits', 'InceptionResnetV2/Logits/Dropout']   
        variables_to_restore = slim.get_variables_to_restore(exclude=excluding)
        pred = tf.nn.softmax(logits)

        #Defining losses and regulization ops:
        with tf.name_scope("loss_op"):
            loss = tf.losses.softmax_cross_entropy(onehot_labels = oh_labels, logits = logits)
        
            total_loss = tf.losses.get_total_loss()#obtain the regularization losses as well
        
        #Create the global step for monitoring the learning_rate and training:
        global_step = tf.train.get_or_create_global_step()

        with tf.name_scope("learning_rate"):    
            lr = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                    global_step=global_step,
                                    decay_steps=decay_steps,
                                    decay_rate = learning_rate_decay_factor,
                                    staircase=True)

    #Define Optimizer with decay learning rate:
        with tf.name_scope("optimizer"):
            optimizer = tf.train.AdamOptimizer(learning_rate = initial_learning_rate)      
            train_op = slim.learning.create_train_op(total_loss,optimizer,
                                                        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        #FIXME: Replace classifier function (sigmoid / softmax)
        with tf.name_scope("metrics"):
            predictions = tf.argmax(pred, 1)
            labels = tf.argmax(oh_labels,1)
            names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': tf.metrics.accuracy(labels, predictions),
            'Precision': tf.metrics.precision(labels, predictions),
            'Recall': tf.metrics.recall(labels, predictions),
            'AUC': tf.metrics.auc(labels,predictions)
            })
            for name, value in names_to_values.items():
                summary_name = 'train/%s' % name
                op = tf.summary.scalar(summary_name, value, collections=[])
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            #Default accuracy
            accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
            # summaries to monitor and group them into one summary op.#
            tf.summary.scalar('accuracy_perso', accuracy)
            tf.summary.scalar('losses/Total_Loss', total_loss)
            tf.summary.scalar('learning_rate', lr)
            tf.summary.scalar('global_step', global_step)
            tf.summary.histogram('proba_perso',pred)        
            #Create the train_op#.
        with tf.name_scope("merge_summary"):       
            my_summary_op = tf.summary.merge_all()
            #Define max steps:
        max_step = num_epochs*num_steps_per_epoch
        #NOTE: We define in this section the properties of the session to run (saver, summaries)
        ckpt_state = tf.train.get_checkpoint_state(train_dir)
        if ckpt_state and ckpt_state.model_checkpoint_path:
            ckpt = ckpt_state.model_checkpoint_path
            saver_b = tf.train.Saver()
        else:
            ckpt = checkpoint_file
            saver_b = tf.train.Saver(variables_to_restore,name="Restoring_Saver")
        #Extracting global variables collections and feed it to our Model Saver
        variables_to_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

if __name__ == '__main__':
    run()