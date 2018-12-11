import tensorflow as tf


from tensorflow.python.platform import tf_logging as logging
import research.slim.nets.nets_factory as nets_factory
from utils.gen_tfrec import load_batch, get_dataset_multiclass, load_batch_dense, load_batch_estimator

import os
import sys
from yaml import load, dump
slim = tf.contrib.slim

#Open and read the yaml file:
stream = open(os.path.join(os.getcwd(), "yaml","config_multilabel.yaml"))
data = load(stream)
stream.close()
#=======Dataset Informations=======#
#==================================#
dataset_dir = data["dataset_dir"]
model_name = data["model_name"]
train_dir = os.path.join(os.getcwd(), "train")
gpu_p = data["gpu_p"]
#Emplacement du checkpoint file
checkpoint_dir= data["checkpoint_dir"]
checkpoint_pattern = data["checkpoint_pattern"]
checkpoint_file = os.path.join(checkpoint_dir, checkpoint_pattern)
ckpt_state = tf.train.get_checkpoint_state(train_dir)
image_size = data["image_size"]
#Nombre de classes à prédire
file_pattern = data["file_pattern"]
file_pattern_for_counting = data["file_pattern_for_counting"]
num_samples = data["num_samples"]
#Création d'un dictionnaire pour reférer à chaque label
#MURA Labels
names_to_labels = data["names_to_labels"]
labels_to_names = data["labels_to_names"]
#==================================#
#=======Training Informations======#
#Nombre d'époques pour l'entraînement
num_epochs = data["num_epochs"]
#State your batch size
batch_size = data["batch_size"]
#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = data["initial_learning_rate"]
#Decay factor
learning_rate_decay_factor = data["learning_rate_decay_factor"]
num_epochs_before_decay = data["num_epochs_before_decay"]
weight_decay = data["weight_decay"]
bn_decay = data["bn_decay"]
stddev = data["stddev"]
#Calculus of batches/epoch, number of steps after decay learning rate
num_batches_per_epoch = int(num_samples / batch_size)
#num_batches = num_steps for one epcoh
decay_steps = int(num_epochs_before_decay * num_batches_per_epoch)
#==================================#
#Create log_dir:
if not os.path.exists(train_dir):
    os.mkdir(os.path.join(os.getcwd(),train_dir))

#===================================================================== Training ===========================================================================#
#Adding the graph:
#Set the verbosity to INFO level
tf.reset_default_graph()
tf.logging.set_verbosity(tf.logging.DEBUG)

def input_fn(mode, dataset_dir,file_pattern, file_pattern_for_counting, labels_to_name, batch_size, image_size):
    train_mode = mode==tf.estimator.ModeKeys.TRAIN
    with tf.name_scope("dataset"):
        dataset = get_dataset_multiclass("train" if train_mode else "eval",
                                        dataset_dir, file_pattern=file_pattern,
                                        file_pattern_for_counting=file_pattern_for_counting,
                                        labels_to_name=labels_to_name)
    with tf.name_scope("load_data"):
        dataset = load_batch_estimator(dataset, model_name, batch_size, image_size, image_size,
                                                        shuffle=train_mode, is_training=train_mode)
    return dataset 

def model_fn(features, mode):
    train_mode = mode==tf.estimator.ModeKeys.TRAIN
    tf.summary.histogram("final_image_hist", features['image/encoded'])
    #Create the model structure using network_fn :
    network = nets_factory.networks_map[model_name]
    network_argscope = nets_factory.arg_scopes_map[model_name]
    with slim.arg_scope(network_argscope(is_training=train_mode, weight_decay=weight_decay, stddev=stddev, bn_decay=bn_decay)):
        #TODO: Check mobilenet_v1 module, var "excluding
        logits, _ = network (features['image/encoded'], num_classes = len(labels_to_names))
    excluding = ['MobilenetV2/Logits']   
    variables_to_restore = slim.get_variables_to_restore(exclude=excluding)
    if (not ckpt_state) and checkpoint_file and train_mode:
        variables_to_restore = variables_to_restore[1:]
        tf.train.init_from_checkpoint(checkpoint_file, 
                            {v.name.split(':')[0]: v for v in variables_to_restore})
   
    
    if mode != tf.estimator.ModeKeys.PREDICT:
         #Defining losses and regulization ops:
        with tf.name_scope("loss_op"):
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = features['image/class/id'], logits = logits)
            total_loss = tf.losses.get_total_loss() #obtain the regularization losses as well
        #TODO: Add a func to transform logit tensor to a label-like tensor
        # If value[][class_id]<0.5 then value[][class_id] = 0. else value[][class_id]= 1.
        #It is necessary for a multilabel classification problem
        logits_sig = tf.nn.sigmoid(logits,name="Sigmoid")
        logits_sig = tf.to_float(tf.to_int32(logits_sig>=0.5))
      
        metrics = {
            'Accuracy': tf.metrics.accuracy(features['image/class/id'], logits_sig, name="acc_op"),
            'Precision': tf.metrics.precision(features['image/class/id'], logits_sig, name="precision_op"),
            'Recall': tf.metrics.recall(features['image/class/id'], logits_sig, name="recall_op"),
            'Acc_Class': tf.metrics.mean_per_class_accuracy(features['image/class/id'], logits_sig, len(labels_to_names), name="per_class_acc")
        }
        for name, value in metrics.items():
            items_list = value[1].get_shape().as_list()
            if len(items_list) != 0:
                for k in range(items_list[0]):
                    tf.summary.scalar(name+"_"+labels_to_names[str(k)], value[1][k])
            else:
                tf.summary.scalar(name, value[1])
        #For Evaluation Mode
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)
        else:
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
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)
    

    #For Predict/Inference Mode:
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes':logits,
            'probabilities': tf.nn.sigmoid(logits, name="Sigmoid")
            }      
        export_outputs = {
        'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        
        return tf.estimator.EstimatorSpec(mode,predictions=predictions,
                                            export_outputs=export_outputs)
def main():
    #Define the checkpoint state to determine initialization: from pre-trained weigths or recovery
           
    #Define max steps:
    max_step = num_epochs*num_batches_per_epoch
    #Define configuration non-distributed work:
    run_config = tf.estimator.RunConfig(model_dir=train_dir, save_checkpoints_steps=num_batches_per_epoch,keep_checkpoint_max=num_epochs)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(tf.estimator.ModeKeys.TRAIN,
                                                dataset_dir,file_pattern,
                                                file_pattern_for_counting, names_to_labels,
                                                batch_size, image_size),max_steps=max_step)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(tf.estimator.ModeKeys.EVAL,
                                                    dataset_dir, file_pattern,
                                                    file_pattern_for_counting, names_to_labels,
                                                    batch_size,image_size))
    work = tf.estimator.Estimator(model_fn = model_fn,
                                    model_dir=train_dir,
                                    config=run_config)
       
    tf.estimator.train_and_evaluate(work, train_spec, eval_spec)
if __name__ == '__main__':
    main()