import tensorflow as tf
import os
from research.slim.preprocessing import inception_preprocessing
import DenseNet.preprocessing.densenet_pre as dp
slim = tf.contrib.slim

items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either... ....',
    'label': 'A label that is as such -- fruits'
}

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

    labels_map = labels_to_name
    num_class = len(labels_to_name)
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

def load_batch(dataset, batch_size, height, width,is_training=True):

    """ Fucntion for loading a train batch 
    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
    """

    #First, create a provider given by slim:
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3*batch_size,
        common_queue_min = 24,
        shuffle = True
    )

    raw_image, true_height, true_width, label = provider.get(['image','height','width','label'])
    raw_image = tf.image.convert_image_dtype(raw_image, dtype=tf.float32)  #Preprocessing using inception_preprocessing:
    #Invert true_height and true_width to tf.int32 required by preprocess_image
    image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)


    one_hot_labels = tf.cast(tf.one_hot(label, depth=dataset.num_classes, on_value=1.0, off_value = 0.0), tf.int64)

    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, one_hot_labels, labels = tf.train.shuffle_batch(
        [image, raw_image, one_hot_labels, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        min_after_dequeue = batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, one_hot_labels, labels

def load_batch_dense(dataset, batch_size, height, width,is_training=True):

    """ Function for loading a train batch 
    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).
    """

    #First, create a provider given by slim:
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3*batch_size,
        common_queue_min = 24,
    )

    raw_image, true_height, true_width, label = provider.get(['image','height','width','label'])
    raw_image = tf.image.convert_image_dtype(raw_image, dtype=tf.float32)
    #Preprocessing using inception_preprocessing:
   
    image = dp.preprocess_image(raw_image, height, width, is_training)
    

    one_hot_labels = tf.cast(tf.one_hot(label, depth=dataset.num_classes, on_value=1.0, off_value=0.0), tf.int64)

    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, one_hot_labels, labels = tf.train.shuffle_batch(
        [image, raw_image, one_hot_labels, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        min_after_dequeue = batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, one_hot_labels, labels