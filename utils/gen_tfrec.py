import tensorflow as tf
import os
import research.slim.preprocessing.preprocessing_factory as preprocessing_factory
import DenseNet.preprocessing.densenet_pre as dp
slim = tf.contrib.slim

#NOTE: This code has to be upgraded: false images transmitted to network

def get_dataset(phase_name, dataset_dir, file_pattern, file_pattern_for_counting, labels_to_name):
    """Creates dataset based on phased_name(train or evaluation), datatset_dir. """

    #On vérifie si phase_name est 'train' ou 'validation'
    if phase_name not in ['train', 'eval']:
        raise ValueError('The phase_name %s is not recognized. Please input either train or eval as the phase_name' % (phase_name))
    #TODO: Remove counting num_samples. num_samples have to be fixed before
    #Compte le nombre total d'examples dans tous les fichiers
    file_pattern_for_counting = file_pattern_for_counting + '_' + phase_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    dataset = tf.data.TFRecordDataset(tfrecords_to_count)
    def parse_fn(example):
        #Create the keys_to_features dictionary for the decoder    
        feature = {
            'image/encoded':tf.FixedLenFeature((), tf.string),
            'image/class/id':tf.FixedLenFeature((), tf.int64),
        }
        parsed_example = tf.parse_single_example(example, feature)
        parsed_example['image/encoded'] = tf.image.decode_image(parsed_example['image/encoded'], channels=3)
        parsed_example['image/encoded'] = tf.image.convert_image_dtype(parsed_example['image/encoded'], dtype=tf.float32)
        return parsed_example
    dataset = dataset.map(parse_fn, num_parallel_calls=8)
    return dataset

def get_dataset_multiclass(phase_name, dataset_dir, file_pattern, file_pattern_for_counting, labels_to_name):
    """Creates dataset based on phased_name(train or evaluation), datatset_dir. """

    #On vérifie si phase_name est 'train' ou 'validation'
    if phase_name not in ['train', 'eval']:
        raise ValueError('The phase_name %s is not recognized. Please input either train or eval as the phase_name' % (phase_name))

    #TODO: Remove counting num_samples. num_samples have to be fixed before
    #Compte le nombre total d'examples dans tous les fichiers
    file_pattern_for_counting = file_pattern_for_counting + '_' + phase_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
    dataset = tf.data.TFRecordDataset(tfrecords_to_count)
    num_class = len(labels_to_name)
    def parse_fn(example):
        #Create the keys_to_features dictionary for the decoder    
        feature = {
            'image/encoded':tf.FixedLenFeature((), tf.string),
            'image/class/id':tf.FixedLenFeature([num_class], tf.float32),
        }
        parsed_example = tf.parse_single_example(example, feature)
        parsed_example['image/encoded'] = tf.image.decode_image(parsed_example['image/encoded'], channels = 3)
        parsed_example['image/encoded'] = tf.image.convert_image_dtype(parsed_example['image/encoded'], dtype = tf.float32)
        return parsed_example
    dataset = dataset.map(parse_fn)
    return dataset

def load_batch(dataset, batch_size, height, width, num_epochs=-1, is_training=True, shuffle=True):

    """ Fucntion for loading a train batch 
    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
    """
    dataset = dataset.repeat(num_epochs)
    def process_fn(example):
        example['image/encoded'].set_shape([None,None,3])
        example['image/encoded']= inception_preprocessing.preprocess_image(example['image/encoded'], height, width, is_training)
        return example
    
    dataset = dataset.map(process_fn)
    if shuffle:
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=True)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    parsed_batch = dataset.make_one_shot_iterator().get_next()

    return parsed_batch['image/encoded'], parsed_batch['image/class/one_hot']

def load_batch_dense(dataset, batch_size, height, width, num_epochs=-1, is_training=True, shuffle=True):
    """ Function for loading a train batch 
    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
    """
    def process_fn(example):
        example['image/encoded'].set_shape([None,None,3])
        example['image/encoded'] = dp.preprocess_image(example['image/encoded'], height, width, is_training)
        
        return example
    dataset = dataset.map(process_fn)
    if shuffle:
        dataset = dataset.shuffle(1000)
    if is_training:
        dataset = dataset.repeat(num_epochs)
    else:
        dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size)
    parsed_batch = dataset.make_one_shot_iterator().get_next()
    return parsed_batch['image/encoded'], parsed_batch['image/class/one_hot']

def load_batch_estimator(dataset, model_name, batch_size, height, width, num_epochs=-1, is_training=True, shuffle=True):
    """ Function for loading a train batch 
    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
    """
    def process_fn(example):
        example['image/encoded'].set_shape([None,None,3])
        #Use model_name and slim to get the preprocessing fucntion :
        preprocess_func = preprocessing_factory.get_preprocessing(model_name, is_training=is_training)
        example['image/encoded'] = preprocess_func(example['image/encoded'], height, width)
        return example
    
    dataset = dataset.map(process_fn, num_parallel_calls=16)
    if is_training and shuffle:
        dataset = dataset.shuffle(3000)
        dataset = dataset.repeat(-1)
    #Batch up the dataset
    dataset = dataset.batch(batch_size)
    #The following line try to minimize the bottleneck btw CPU and GPU
    #prefecth always prepare a amount of data on CPU for the GPU
    dataset = dataset.prefetch(batch_size)
    return dataset