import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
import research.slim.nets.mobilenet_v1 as mobilenet_v1
import research.slim.datasets.imagenet as imagenet
from research.slim.preprocessing import inception_preprocessing

import os
import time
import datetime

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('dataset_dir',None,'String: Your dataset directory')
flags.DEFINE_string('eval_dir', None, 'String: Your train directory')

FLAGS = flags.FLAGS
#=======Dataset Informations=======#
dataset_dir = FLAGS.dataset_dir

log_dir="./log_eval"

log_eval = './train_fruit/log_eval'
#Emplacement du checkpoint file
checkpoint_dir = "./model.ckpt-36480"
image_size = 224
#Nombre de classes à prédire
num_class = 65

file_pattern = "fruit360_%s_*.tfrecord"
file_pattern_for_counting = "fruit360"
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
num_epochs = 1

#State your batch size
batch_size = 16




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


def load_batch(dataset, batch_size, height=image_size, width=image_size,is_training=False):

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
    image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)
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
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

    with tf.Graph().as_default():
    #=========== Evaluate ===========#
        global_step_cs = tf.train.get_or_create_global_step()
        # Adding the graph:

        dataset = get_dataset("validation", dataset_dir, file_pattern=file_pattern)
        images,_, oh_labels, labels = load_batch(dataset, batch_size)

        #Calcul of batches/epoch, number of steps after decay learning rate
        num_batches_per_epoch = int(dataset.num_samples / batch_size)
        num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed


        #Create the model inference
        """with slim.arg_scope(mobilenet_v1.mobilenet_v1_arg_scope(is_training=True)):"""
        logits, end_points = mobilenet_v1.mobilenet_v1_050(images, num_classes = dataset.num_classes, is_training = False)
        variables_to_restore = slim.get_variables_to_restore()
    

    

        #Defining accuracy and regulization ops:
    
        predictions = tf.argmax(end_points['Predictions'], 1)
        labels = tf.squeeze(labels)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))
        probabilities = end_points['Predictions']

        #Define the metrics to evaluate
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(logits, labels, 5),
        })

        #Define and merge summaries:
        tf.summary.scalar('Accuracy', accuracy)
        tf.summary.histogram('Predictions', probabilities)
        summary_op = tf.summary.merge_all()

        #This is the common way to define evaluation using slim
        max_step = num_epochs*num_steps_per_epoch
        initial_op=tf.group(tf.global_variables_initializer(),
                        tf.local_variables_initializer())

        slim.evaluation.evaluate_once(
            master = '',  
            checkpoint_path = checkpoint_dir,
            logdir = log_dir,
            num_evals = max_step,
            initial_op = initial_op,
            eval_op = list(names_to_updates.values()),
            summary_op = summary_op,
            variables_to_restore = variables_to_restore)

if __name__ == '__main__':
    run()