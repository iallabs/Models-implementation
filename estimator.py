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
flags.DEFINE_string('dataset_dir','','String: Your dataset directory')
flags.DEFINE_string('train_dir', 'train_fruit/training', 'String: Your train directory')
flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
flags.DEFINE_string('ckpt','train_fruit/net/mobilenet_v1_0.5_160.ckpt','String: Your dataset directory')
FLAGS = flags.FLAGS

#=======Dataset Informations=======#
#==================================#
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
num_samples = 36807
#Création d'un dictionnaire pour reférer à chaque label
labels_to_name = {
    'negative':0,
    'positive':1
}
#==================================#
#=======Training Informations======#
#Nombre d'époques pour l'entraînement
num_epochs = 100
#State your batch size
batch_size = 16
#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 1e-4
#Decay factor
learning_rate_decay_factor = 0.95
num_epochs_before_decay = 1
#Calculus of batches/epoch, number of steps after decay learning rate
num_batches_per_epoch = int(num_samples / batch_size)
#num_batches = num_steps for one epcoh
decay_steps = int(num_epochs_before_decay * num_batches_per_epoch)
#==================================#
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

def input_fn(mode, dataset_dir,file_pattern, file_pattern_for_counting, labels_to_name, batch_size, image_size):
    train_mode = mode==tf.estimator.ModeKeys.TRAIN
    with tf.name_scope("dataset"):
        dataset = get_dataset("train" if train_mode else "validation",
                                        dataset_dir, file_pattern=file_pattern,
                                        file_pattern_for_counting=file_pattern_for_counting,
                                        labels_to_name=labels_to_name)
    with tf.name_scope("load_data"):
        dataset = load_batch_dense(dataset, batch_size, image_size, image_size, num_epochs,
                                                        shuffle=train_mode, is_training=train_mode)
    return dataset 

def model_fn(images, onehot_labels, mode, num_classes, checkpoint_state):
    train_mode = mode==tf.estimator.ModeKeys.TRAIN
    print("training_mode:"+str(train_mode))
    print(mode)
    #Create the model inference
    with slim.arg_scope(mobilenet_v2.training_scope(is_training=train_mode, weight_decay=0.0005, stddev=1., bn_decay=0.97)):
            #TODO: Check mobilenet_v1 module, var "excluding
            logits, _ = mobilenet_v2.mobilenet(images,depth_multiplier=1.0, num_classes = len(labels_to_name))
    predictions = {
            'classes':tf.argmax(logits, axis=1),
            'probabilities': tf.nn.softmax(logits, name="Softmax")
        }
    #For Predict/Inference Mode:
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)    
    
    excluding = ['MobilenetV2/Logits']   
    variables_to_restore = slim.get_variables_to_restore(exclude=excluding)
    if not checkpoint_state and checkpoint_file:
        tf.train.init_from_checkpoint(checkpoint_file, 
                            {v.name.split(':')[0]: v for v in variables_to_restore})
    #Defining losses and regulization ops:
    with tf.name_scope("loss_op"):
        loss = tf.losses.softmax_cross_entropy(onehot_labels = onehot_labels, logits = logits)
        total_loss = tf.losses.get_total_loss()#obtain the regularization losses as well
    #FIXME: Replace classifier function (sigmoid / softmax)
    with tf.name_scope("metrics"):
        pred = predictions['classes']
        labels = tf.argmax(onehot_labels, 1)
        metrics = {
        'Accuracy': tf.metrics.accuracy(labels, pred),
        'Precision': tf.metrics.precision(labels, pred),
        'Recall': tf.metrics.recall(labels, pred),
        'AUC': tf.metrics.auc(labels,pred)
        }
        for name, value in metrics.items():
            tf.summary.scalar(name, value[1])
        tf.summary.histogram('proba_perso',predictions['probabilities'])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions, loss=loss, eval_metric_ops=metrics)

    #Create the global step for monitoring the learning_rate and training:
    global_step = tf.train.get_or_create_global_step()
    with tf.name_scope("learning_rate"):    
        lr = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                global_step=global_step,
                                decay_steps=decay_steps,
                                decay_rate = learning_rate_decay_factor,
                                staircase=True)
        tf.summary.scalar('learning_rate', lr)
    #Define Optimizer with decay learning rate:
    with tf.name_scope("optimizer"):
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)      
        train_op = slim.learning.create_train_op(total_loss,optimizer,
                                                    update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    return tf.estimator.EstimatorSpec(mode, predictions=predictions, loss=total_loss, train_op=train_op)

def main():
    #Define the checkpoint state to determine initialization: from pre-trained weigths or recovery
    ckpt_state = tf.train.get_checkpoint_state(train_dir)       
    #Define max steps:
    max_step = num_epochs*num_batches_per_epoch
    #Define configuration non-distributed work:
    run_config = tf.estimator.RunConfig(model_dir=train_dir, save_checkpoints_steps=num_batches_per_epoch)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(tf.estimator.ModeKeys.TRAIN,
                                                dataset_dir,file_pattern,
                                                file_pattern_for_counting, labels_to_name,
                                                batch_size, image_size),max_steps=max_step)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(tf.estimator.ModeKeys.EVAL,
                                                    dataset_dir, file_pattern,
                                                    file_pattern_for_counting, labels_to_name,
                                                    batch_size,image_size))
    work = tf.estimator.Estimator(model_fn = lambda features,labels, mode, config: model_fn(features, labels, mode, len(labels_to_name), ckpt_state),
                                model_dir=train_dir,
                                config=run_config)
       
    tf.estimator.train_and_evaluate(work, train_spec, eval_spec)
if __name__ == '__main__':
    main()