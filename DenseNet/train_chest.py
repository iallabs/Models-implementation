import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging

import nets.densenet as densenet

from research.slim.preprocessing import vgg_preprocessing

import os
import time
import datetime

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('dataset_dir',"data",'String: Your dataset directory')
flags.DEFINE_string('train_dir', "trainlogs", 'String: Your train directory')
flags.DEFINE_boolean('log_device_placement', True,
                            """Whether to log device placement.""")
flags.DEFINE_string('ckpt',"ckpt/tf-densenet121.ckpt",'String: Your dataset directory')
FLAGS = flags.FLAGS

#=======Dataset Informations=======#
dataset_dir = FLAGS.dataset_dir

#Emplacement du checkpoint file
checkpoint_file= FLAGS.ckpt

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
#Nombre d'époques pour l'entraîen
num_epochs = 30

#State your batch size
batch_size = 16

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.001
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 3

#We now create a function that creates a Dataset class which will give us many TFRecord files to feed in the examples into a queue in parallel.
def get_dataset(phase_name, dataset_dir, file_pattern=file_pattern, file_pattern_for_counting=file_pattern_for_counting):
    """Creates dataset based on phased_name(train or validation), datatset_dir. """
    
    #On vérifie si phase_name est 'train' ou 'validation'
    if phase_name not in ['train', 'validation']:
        raise ValueError('The phase_name %s is not recognized. Please input either train or validation as the phase_name' % (phase_name))

    file_pattern_path = os.path.join(dataset_dir, file_pattern%(phase_name))

    #Compte le nombre total d'examples dans tous les fichiers
    num_samples = 0
    file_pattern_for_counting = file_pattern_for_counting + '_' + phase_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    #Création d'un "reader", de type TFrecord pour ce cas précis:
    reader = tf.TFRecordReader

    #Create the keys_to_features dictionary for the decoder    
    feature = {
        'image/encoded':tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label':tf.FixedLenFeature((), tf.int64,default_value=tf.zeros([], dtype=tf.int64)),
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    #Decoder, provided by slim
    decoder = slim.tfexample_decoder.TFExampleDecoder(feature, items_to_handlers)

    labels_map= labels_to_name

    #create the dataset:
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = num_class,
        labels_to_name = labels_map,
        items_to_descriptions = items_to_descriptions)
    
    return dataset

def load_batch(dataset, batch_size, height=image_size, width=image_size,is_training=True):

    """ Fucntion for loading a train batch 
    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
    """

    #First, create a provider given by slim:
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3*batch_size,
        common_queue_min = 24
    )

    raw_image, label = provider.get(['image','label'])
    #Preprocessing using inception_preprocessing:
    image = vgg_preprocessing.preprocess_image(raw_image, height, width, is_training)
    one_hot_labels = slim.one_hot_encoding(label, dataset.num_classes)
    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, one_hot_labels, labels = tf.train.batch(
        [image, raw_image, one_hot_labels, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, one_hot_labels, labels

def run():
    #Create log_dir:
    if not os.path.exists(FLAGS.train_dir):
        os.mkdir(FLAGS.train_dir)

    #=========== Training ===========#
    #Adding the graph:
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

        dataset = get_dataset("train", dataset_dir, file_pattern=file_pattern)
        images,_, oh_labels, labels = load_batch(dataset, batch_size)

        #Calcul of batches/epoch, number of steps after decay learning rate
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
        decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

        #Create the model inference
        """with slim.arg_scope(densenet.densenet_arg_scope(is_training=True)):"""
        logits, end_points = densenet.densenet121(images, num_classes = dataset.num_classes, is_training = True)

        excluding = ['densenet121/Logits']
        variable_to_restore = slim.get_variables_to_restore(exclude=excluding)
        slim.assign_from_checkpoint(checkpoint_file, variable_to_restore)


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

        #Create the train_op.
        train_op = slim.learning.create_train_op(total_loss, optimizer)


        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        gen_acc , accuracy_update = tf.metrics.accuracy(labels, predictions)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
        metrics_op = tf.group(accuracy_update, probabilities)

        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        tf.summary.scalar('gen_accuracy', gen_acc)
        tf.summary.histogram('probabilities', probabilities)
        my_summary_op = tf.summary.merge_all()

        

        class _LoggerHook(tf.train.SessionRunHook):

            def begin(self):
                self._step = global_step_init

            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs([total_loss, accuracy])  # Asks for loss value.

            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value, accuracy_value = run_values.results
                if self._step % 1 == 0:
                    num_examples_per_step = batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f, accuracy=%.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print (format_str % (datetime.time(), self._step, loss_value, accuracy_value,
                               examples_per_sec, sec_per_batch))



        max_step = num_epochs*num_steps_per_epoch

        saver = tf.train.Saver(variable_to_restore)
      
        #Define your supervisor for running a managed session:
        supervisor = tf.train.MonitoredTrainingSession(checkpoint_dir=FLAGS.train_dir,
                                                        hooks=[tf.train.StopAtStepHook(last_step=max_step),
                                                                tf.train.NanTensorHook(loss),
                                                                _LoggerHook()],
                                                        config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement),
                                                        save_checkpoint_secs=300,
                                                        save_summaries_steps=100)

        #Running session:
        with supervisor as sess:
            ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                saver.restore(sess, checkpoint_file)
            while not sess.should_stop():
                sess.run(train_op)

if __name__ == '__main__':
    run()