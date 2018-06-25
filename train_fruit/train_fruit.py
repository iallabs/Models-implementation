import tensorflow as tf

from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging

import research.slim.nets.mobilenet_v1 as mobilenet_v1
import research.slim.datasets.imagenet as imagenet
from research.slim.preprocessing import inception_preprocessing

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

items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either... ....',
    'label': 'A label that is as such -- fruits'
}

#=======Training Informations======#
#Nombre d'époques pour l'entraîen
num_epochs = 35

#State your batch size
batch_size = 16

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0005
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
        'image/height': tf.FixedLenFeature((), tf.int64),
        'image/width': tf.FixedLenFeature((), tf.int64),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label':tf.FixedLenFeature((), tf.int64,default_value=tf.zeros([], dtype=tf.int64)),
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'height':slim.tfexample_decoder.Tensor('image/height'),
        'width': slim.tfexample_decoder.Tensor('image/width'),
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

    raw_image, true_height, true_width, label = provider.get(['image','height','width','label'])

    #Preprocessing using inception_preprocessing:
    #Invert true_height and true_width to tf.int32 required by preprocess_image
    image = inception_preprocessing.preprocess_image(raw_image, tf.cast(true_height,tf.int32), tf.cast(true_width, tf.int32), is_training)
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_nearest_neighbor(image,[height,width])
    image = tf.squeeze(image)

    one_hot_labels = tf.cast(tf.one_hot(label, depth=dataset.num_classes), tf.int64)

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
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

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
        """with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):"""
        logits, end_points = mobilenet_v1.mobilenet_v1_050(images, num_classes = dataset.num_classes, is_training = True)

        excluding = ['MobilenetV1/Logits', 'MobilenetV1/AuxLogits']
        variable_to_restore = slim.get_variables_to_restore(exclude=excluding)
        init_assign_op, init_feed_dict = slim.assign_from_checkpoint(checkpoint_file, variable_to_restore)


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
        
        

        #State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
            'Accuracy': tf.metrics.accuracy(labels, predictions),
            'True_positives': tf.metrics.true_positives(labels, predictions),
            'False_positives': tf.metrics.false_positives(labels, predictions),
            'False_negatives': tf.metrics.false_negatives(labels, predictions),
            'True_negatives': tf.metrics.true_negatives(labels, predictions),
            })

        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)


        train_op = slim.learning.create_train_op(total_loss, optimizer)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

        #Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        tf.summary.histogram('probabilities', probabilities)
        my_summary_op = tf.summary.merge_all()
       
        def InitAssignFn(sess):
            sess.run(init_assign_op, init_feed_dict)

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


        max_step = num_epochs*num_steps_per_epoch

        saver = tf.train.Saver(variable_to_restore)

        class _scaffold(tf.train.Scaffold):
            def __init__(self):
                super().__init__(summary_op=my_summary_op, init_fn=init_assign_op)
      
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