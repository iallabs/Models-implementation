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
flags.DEFINE_string('dataset_dir','D:/fruits/fruits-360','String: Your dataset directory')
flags.DEFINE_string('train_dir', 'train_fruit/training', 'String: Your train directory')
flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
flags.DEFINE_string('ckpt','train_fruit/net/mobilenet_v1_0.5_160.ckpt','String: Your dataset directory')
FLAGS = flags.FLAGS

#=======Dataset Informations=======#
dataset_dir = FLAGS.dataset_dir

log_dir="log"

#Emplacement du checkpoint file
checkpoint_file= FLAGS.ckpt

image_size = 224
#Nombre de classes à prédire

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


#=======Training Informations======#
#Nombre d'époques pour l'entraînement
num_epochs = 35

#State your batch size
batch_size = 16

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0005
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 1

def run():
    #Create log_dir:
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    #===================================================================== Training ===========================================================================#
    #Adding the graph:
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

        dataset = get_dataset("train", dataset_dir, file_pattern=file_pattern, file_pattern_for_counting=file_pattern_for_counting, labels_to_name=labels_to_name)
        images,_, oh_labels, labels = load_batch_dense(dataset, batch_size, image_size, image_size)

        #Calcul of batches/epoch, number of steps after decay learning rate
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        #Create the model inference
        """with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):"""
        #TODO: Check mobilenet_v1 module, var "excluding
        net, end_points = mobilenet_v1.mobilenet_v1_050(images, num_classes = None, is_training = True)

        excluding = ['MobilenetV1/Logits', 'MobilenetV1/AuxLogits']
        variable_to_restore = slim.get_variables_to_restore(exclude=excluding)
        """init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_file, variable_to_restore)"""

        #We reconstruct a FCN block on top of our final conv layer. 
        net = slim.dropout(net, keep_prob=0.5, scope='Dropout_1b')
        net = slim.conv2d(net, 512, [1,1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1')
        net = slim.dropout(net, keep_prob=0.5, scope='Dropout_1b')
        net = slim.conv2d(net, 256, [1,1], activation_fn=None, normalizer_fn=None, scope='Conv2d_1c_1x1_1')
        logits = slim.conv2d(net, dataset.num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1_2')
        logits = tf.nn.relu(logits, name='final_relu')
        logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        end_points['Predictions'] = logits

        #Defining losses and regulization ops:
        loss = tf.losses.softmax_cross_entropy(onehot_labels = oh_labels, logits = logits)
        total_loss = tf.losses.get_total_loss()    #obtain the regularization losses as well
        
        #Create the global step for monitoring the learning_rate and training:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        global_step_init = -1
        if ckpt and ckpt.model_checkpoint_path:
            global_step_init = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
            global_step = tf.Variable(global_step_init, name='global_step', dtype=tf.int64, trainable=False)

        else:
            global_step = tf.train.get_or_create_global_step()

        lr = tf.train.exponential_decay(learning_rate=initial_learning_rate,
                                        global_step=global_step,
                                        decay_steps=decay_steps,
                                        decay_rate = learning_rate_decay_factor,
                                        staircase=True)

        #Define Optimizer with decay learning rate:
        optimizer = tf.train.AdamOptimizer(learning_rate = lr)      

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': tf.metrics.accuracy(labels, predictions),
            })

        for name, value in names_to_values.items():
            summary_name = 'train/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        #Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('accuracy_perso', accuracy)
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('learning_rate', lr)
        tf.summary.histogram('probabilities', probabilities)
        my_summary_op = tf.summary.merge_all()
        #Define max steps:
        max_step = num_epochs*num_steps_per_epoch
        #Create a saver to load pre-trained model
        saver = tf.train.Saver(variable_to_restore)

        #Create a class Hook for your training. Handles the prints and ops to run
        #  
        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self._step = global_step_init
                self.totalloss=0.0
                self.totalacc=0.0

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs([total_loss, accuracy, names_to_updates])  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value, accuracy_value, update = run_values.results
                if self._step % 1 == 0:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)
                    self.totalloss += loss_value
                    self.totalacc += accuracy_value
                    format_str = ('\r%s: step %d, avgloss = %.2f, loss = %.2f, avgacc= %.2f ,accuracy=%.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    sys.stdout.write(format_str % (datetime.time(), self._step, self.totalloss/self._step, loss_value, self.totalacc/self._step, accuracy_value,
                               examples_per_sec, sec_per_batch))



        #Define your supervisor for running a managed session:
        supervisor = tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
                                                        hooks=[tf.train.StopAtStepHook(last_step=max_step),
                                                                tf.train.NanTensorHook(loss),
                                                                _LoggerHook()],
                                                        config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement),
                                                        save_checkpoint_secs=300,
                                                        save_summaries_steps=100)
        i = 0
        #Running session:
        with supervisor as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                saver.restore(sess, checkpoint_file)
            while not sess.should_stop():
                if (i+1) % num_steps_per_epoch == 0:
                    ckpt_eval = tf.train.get_checkpoint_state(FLAGS.train_dir)
                    evaluate(ckpt_eval.model_checkpoint_path,
                             dataset_dir,
                             file_pattern,
                             file_pattern_for_counting,
                             labels_to_name,
                             batch_size,
                             image_size,
                            )
                sess.run(train_op)
                i += 1

if __name__ == '__main__':
    run()