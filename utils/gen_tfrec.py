import tensorflow as tf
import os
from research.slim.preprocessing import inception_preprocessing
import DenseNet.preprocessing.densenet_pre as dp
slim = tf.contrib.slim

#NOTE: This code has to be upgraded: false images transmitted to network

def get_dataset(phase_name, dataset_dir, file_pattern, file_pattern_for_counting, labels_to_name):
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

    dataset = tf.data.TFRecordDataset(tfrecords_to_count)
    num_class = len(labels_to_name)
    def parse_fn(example):
        #Create the keys_to_features dictionary for the decoder    
        feature = {
            'image/encoded':tf.FixedLenFeature((), tf.string),
            'image/filename':tf.FixedLenFeature((), tf.string),
            'image/height': tf.FixedLenFeature((), tf.int64),
            'image/width': tf.FixedLenFeature((), tf.int64),
            'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
            'image/class/label':tf.FixedLenFeature((), tf.int64),
        }
        parsed_example = tf.parse_single_example(example, feature)
        parsed_example['image/encoded'] = tf.image.decode_image(parsed_example['image/encoded'], channels=3)
        parsed_example['image/encoded'] = tf.image.convert_image_dtype(parsed_example['image/encoded'], dtype=tf.float32)
        labels = parsed_example['image/class/label']
        parsed_example['image/class/one_hot'] = tf.cast(tf.one_hot(labels, depth=num_class, on_value=1.0, off_value = 0.0), tf.int64)

        return parsed_example
    dataset = dataset.map(parse_fn)
    return dataset, num_samples

def load_batch(dataset, batch_size, height, width, num_epochs, is_training=True, shuffle=True):

    """ Fucntion for loading a train batch 
    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
    """
    dataset = dataset.repeat(num_epochs)
    def process_fn(example):
        tf.summary.image("final_image", example['image/encoded'])
        example['image/encoded'].set_shape([None,None,3])
        example['image/encoded']= inception_preprocessing.preprocess_image(example['image/encoded'], height, width, is_training)
        return example
    
    dataset = dataset.map(process_fn)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    parsed_batch = dataset.make_one_shot_iterator().get_next()
    tf.summary.image("final_image", parsed_batch['image/encoded'])

    return parsed_batch['image/encoded'], parsed_batch['image/filename'], parsed_batch['image/class/one_hot'], parsed_batch['image/class/label']

def load_batch_dense(dataset, batch_size, height, width, num_epochs=None, is_training=True, shuffle=True):

    """ Function for loading a train batch 
    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
    """
    def process_fn(example):
        tf.summary.image("final_image", example['image/encoded'])
        example['image/encoded'].set_shape([None,None,3])
        example['image/encoded'] = dp.preprocess_image(example['image/encoded'], height, width, is_training)
        
        return example
    dataset = dataset.map(process_fn)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    parsed_batch = dataset.make_one_shot_iterator().get_next()
    tf.summary.image("final_image", parsed_batch['image/encoded'])
    return parsed_batch['image/encoded'], parsed_batch['image/filename'], parsed_batch['image/class/one_hot'], parsed_batch['image/class/label']