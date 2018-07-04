import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
import nets.densenet as densenet
import preprocessing.densenet_pre as dp
from research.slim.preprocessing import inception_preprocessing

import os
import time
import datetime

slim = tf.contrib.slim

flags = tf.app.flags
flags.DEFINE_string('dataset_dir',None,'String: Your dataset directory')
flags.DEFINE_string('eval_dir', './eval_logs', 'String: Your train directory')

FLAGS = flags.FLAGS
#=======Dataset Informations=======#
dataset_dir = FLAGS.dataset_dir

log_dir=FLAGS.eval_dir

#Emplacement du checkpoint file
checkpoint_dir = "trainlogs/model."
image_size = 224
#Nombre de classes à prédire
num_class = 15

file_pattern = "chest%s_*.tfrecord"
file_pattern_for_counting = "chest"
#Création d'un dictionnaire pour reférer à chaque label
labels_to_name = {
                0:'No Finding', 
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
                14: 'Hernia'
                }


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
    image = dp.preprocess_image(raw_image, height, width, is_training=is_training)
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
        with slim.arg_scope(densenet.densenet_arg_scope(is_training=False))
            logits, end_points = densenet.densenet121(images, num_classes = dataset.num_classes, is_training = False)
        variables_to_restore = slim.get_variables_to_restore()
        
        
        logit = tf.squeeze(logits)
        predictions = tf.squeeze(tf.argmax(end_points['Predictions'], 3))
        
        #Define the metrics to evaluate
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_sparse_recall_at_k(logit, oh_labels, 5),
        'Recall_15': slim.metrics.streaming_sparse_recall_at_k(logit, oh_labels, 15),
        })

        for name, value in names_to_values.items():
            summary_name = 'eval/%s' % name
            op = tf.summary.scalar(summary_name, value, collections=[])
            op = tf.Print(op, [value], summary_name)
            tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

        # merge summaries:

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