import tensorflow as tf
import os

def load_images(filenames_pattern, num_channels, image_extension='jpg'):
    """
    we want to load data into Datasets in order to perform
    different map-fn fucntions preprocessing

    Args:
    filenames_list : A string representing path pattern of each image
    
    Return:
    tf.data.Dataset containing raw image data, decoded and flattened
    """
    #NOTE: shuffle=false prevent mistakes during labeling images
    dataset = tf.data.Dataset.list_files(filenames_pattern, shuffle=False)
    label_dataset = dataset.map(lambda x: extract_label(x, label_pos=-2), num_parallel_calls=100)
    image_dataset = dataset.map(lambda x: extract_image(x, num_channels), num_parallel_calls=100)
    return image_dataset, label_dataset

def extract_label(filename, label_pos=-1):
    """
    Given a dataset of image's filenames, we want to extract the label
    for each, then returns a dataset
    """
    filename_split = tf.string_split([filename], delimiter=os.sep).values
    #NOTE:The Following line is an efficient way of extracting label for MURA
    label = tf.string_split([filename_split[label_pos]], delimiter = "_").values[-1]
    return label

def extract_image(filename, num_channels):
    """
    Given a dataset of image's filenames and num_chennels per image, we
    read, decode and convert the image dtype
    """
    image_raw = tf.read_file(filename)
#    image = tf.image.decode_image(image_raw, num_channels)
#    image = tf.image.convert_image_dtype(image, tf.float32)
    return image_raw

def per_pixel_mean_stddev(dataset, image_size):
    """
    Compute the mean of each pixel over the entire dataset.
    """
    #NOTE: Replace "3" by the number of channels    
    initial_state = tf.constant(0., dtype=tf.float32, shape=[image_size, image_size, 3])
    dataset = dataset.map(lambda x: resize(x, image_size))
    count = dataset.reduce(0, lambda x, _: x+1)
    pixel_sum = dataset.reduce(initial_state, lambda x, y: tf.add(x, y))
    pixel_mean = tf.divide(pixel_sum, tf.to_float(count))
    return pixel_mean, count

def per_channel_mean_stddev(dataset):
    """
    Compute the mean & stddev of each channel for every image.
    """
    def channel_mean_stddev(decoded_image):
        means = tf.reduce_mean(decoded_image, axis=[0,1])
        stddev = tf.sqrt(tf.reduce_mean(tf.square(decoded_image-means), axis=[0,1]))
        return tf.stack([means, stddev])
    per_channel_mean_stddev_dataset = dataset.map(lambda x: channel_mean_stddev(x))
    return per_channel_mean_stddev_dataset

def per_mean_stddev(dataset):
    """
    Compute the mean & stddev of every image.
    """
    def mean_stddev(decoded_image):
        means = tf.reduce_mean(decoded_image)
        stddev = tf.reduce_mean(tf.sqrt(tf.pow(decoded_image-means,2)))
        return tf.stack([means, stddev])
    return dataset.map(lambda x: mean_stddev(x))

def encode_stats(alpha):
    """
    Utility function that encodes Mean per pixel
    depending on height*width of image
    """
    pass

def resize(image, image_size):
    rank_assertion = tf.Assert(
            tf.equal(tf.rank(image), 4),
            ['Rank of image must be equal to 4.'])
    with tf.control_dependencies([rank_assertion]):
        image = tf.image.resize_bilinear(image, [image_size, image_size])[0]
    return image

#NOTE: Use FLAGS or YAML file to define files_pattern, desired number of channels and image_extension
a, b = load_images("D:/chest/images/*.png", 3 ,image_extension='png')
a, count = per_pixel_mean_stddev(a, 300)
#NOTE: Implement Summaries of "per_pixel_mean_stddev" using tf.summary.histogram
#NOTE: To encode stats, we use the new func tf.data.experimental.TFRecordWriter:
#:: writer = tf.data.experimental.TFRecordWriter(filename)
#:: writer.write(dataset)
#NOTE: The above step will be reproduced to encode pre-processed data
