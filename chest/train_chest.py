import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging

import DenseNet.nets.densenet as densenet

from utils.gen_tfrec import load_batch, get_dataset, load_batch_dense

import os
import sys
import time
import datetime

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_float('gpu_p', 1.0, 'Float: allow gpu growth value to pass in config proto')
flags.DEFINE_string('dataset_dir',"data",'String: Your dataset directory')
flags.DEFINE_string('train_dir', "trainlogs", 'String: Your train directory')
flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
flags.DEFINE_string('ckpt',"ckpt/tf-densenet121.ckpt",'String: Your dataset directory')
FLAGS = flags.FLAGS

#=======Dataset Informations=======#
dataset_dir = FLAGS.dataset_dir
train_dir = FLAGS.train_dir
#Emplacement du checkpoint file
checkpoint_file= FLAGS.ckpt
gpu_p = FLAGS.gpu_p
image_size = 224
#Nombre de classes à prédire
num_class = 15

file_pattern = "chest_%s_*.tfrecord"
file_pattern_for_counting = "chest"
#Création d'un dictionnaire pour reférer à chaque label
labels_to_name = {0:'No Finding', 
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
                14: 'Hernia'}
#Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.

items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either... ....',
    'label': 'A label that is as such -- fruits'
}

#=======Training Informations======#
#Nombre d'époques pour l'entraînement
num_epochs = 60

#State your batch size
batch_size = 64

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 1e-3
learning_rate_decay_factor = 0.95
num_epochs_before_decay = 1


def run():
    #Create log_dir:
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(os.getcwd()+'/'+FLAGS.train_dir)

    #=========== Training ===========#
    #Adding the graph:
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

        dataset = get_dataset("train", dataset_dir, file_pattern=file_pattern, file_pattern_for_counting=file_pattern_for_counting, labels_to_name=labels_to_name)
        images, _, labels, oh_labels, _ = load_batch_dense(dataset, batch_size, image_size, image_size, is_training=True)

        #Calcul of batches/epoch, number of steps after decay learning rate
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        #Create the model inference
        with slim.arg_scope(densenet.densenet_arg_scope(is_training=True)):
            logits, end_points = densenet.densenet121(images, num_classes = len(labels_to_name), is_training = True)
        
        excluding = ['densenet121/logits','densenet121/Predictions']
        variables_to_restore = slim.get_variables_to_restore(exclude=excluding)
        logit = tf.squeeze(logits)
        end_points['Pred_sig']=tf.nn.sigmoid(logit)
        #Defining losses and regulization ops:
        with tf.name_scope("loss_op"):
            loss = tf.losses.sigmoid_cross_entropy(multi_class_labels = oh_labels, logits = logit)
            total_loss = tf.reduce_mean(tf.losses.get_total_loss())    #obtain the regularization losses as well
        
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
            optimizer = tf.train.AdamOptimizer(learning_rate = lr)
            #Create the train_op.
            train_op = optimizer.minimize(total_loss, global_step=global_step)
        
        #Create all summaries needed for monitoring the training
        with tf.name_scope("metrics"):
            #State the metrics that you want to predict. We get a predictions
            predictions = tf.argmax(end_points['Pred_sig'],axis=1)
            gt = tf.argmax(oh_labels, axis=1)
            names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
                'Accuracy': tf.metrics.accuracy(gt, predictions),
                })
            for name, value in names_to_values.items():
                summary_name = 'train/%s' % name
                op = tf.summary.scalar(summary_name, value, collections=[])
                tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)
            accuracy = tf.reduce_mean(tf.cast(tf.equal(gt, predictions), tf.float32))
            tf.summary.scalar('losses/Total_Loss', total_loss)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('learning_rate', lr)
            tf.summary.histogram('images', images)
            tf.summary.histogram('probabilities', predictions)
            my_summary_op = tf.summary.merge_all()
        
        ckpt_state = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt_state and ckpt_state.model_checkpoint_path:
            ckpt = ckpt_state.model_checkpoint_path
        else:
            ckpt = checkpoint_file
            saver = tf.train.Saver(variables_to_restore)
            
        #Define a restore function wrapped for scaffold
        def restore_wrap(scaffold, sess):
            if ckpt != checkpoint_file:
                scaffold.saver.restore(sess, ckpt)
            else:
                saver.restore(sess, ckpt)
        #Define a step_fn function to run the train step: Special to monitored training session:
        txt_file = open("Output.txt", "w")
        def step_fn(step_context):
            value, i,a,b,c = step_context.run_with_hooks([train_op, global_step,labels, oh_labels, end_points['Pred_sig']])
            txt_file.write("*****step i***** " + str(i) + "\n" +"labels : "+ str(a) + "\n" + "oh_labels : "+str(b) +\
                            "\n"+"predictions : "+str(c)+"\n")
            return value
        #Create a class Hook for your training. Handles the prints and ops to run
        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self.step = 0
                self.totalloss=0.0
                self.totalacc=0.0

            def before_run(self, run_context):
                self._start_time = time.time()
                return tf.train.SessionRunArgs([global_step,total_loss, accuracy, names_to_updates])  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                self.step,loss_value, accuracy_value, update = run_values.results
                num_examples_per_step = batch_size
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = float(duration)
                self.totalloss += loss_value
                self.totalacc += accuracy_value
                format_str = ('\r%s: step %d,  avg_loss=%.3f, loss = %.2f, avg_auc=%.2f, accuracy=%.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
                sys.stdout.write(format_str % (datetime.time(), self.step+1, self.totalloss/(self.step+1), loss_value, 
                                self.totalacc/(self.step+1), accuracy_value, examples_per_sec, sec_per_batch))

        #deFINE A ConfigProto to allow gpu device
        config = tf.ConfigProto()
        config.log_device_placement = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_p

        #Definine checkpoint path for restoring the model
        max_step = num_epochs*num_steps_per_epoch

        scaffold = tf.train.Scaffold(init_fn= restore_wrap, summary_op=my_summary_op)
        saver_hook= tf.train.CheckpointSaverHook(train_dir,save_steps=num_batches_per_epoch//2,scaffold=scaffold)
        summary_hook = tf.train.SummarySaverHook(save_steps=20, output_dir=train_dir, scaffold=scaffold)
        #Define your supervisor for running a managed session:
        supervisor = tf.train.MonitoredSession( session_creator=tf.train.ChiefSessionCreator(master=None,checkpoint_dir=train_dir,
                                                                                             config=config, scaffold=scaffold),
                                                hooks=[tf.train.StopAtStepHook(last_step=max_step),
                                                        tf.train.NanTensorHook(loss), saver_hook,
                                                        _LoggerHook(), summary_hook])
        
        #Running session:
        with supervisor as sess:
            print(tf.get_collection(tf.GraphKeys.INIT_OP))
            while not sess.should_stop():
                _ = sess.run_step_fn(step_fn)
                
        txt_file.close()

if __name__ == '__main__':
    run()