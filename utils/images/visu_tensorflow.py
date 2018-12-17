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
    image = tf.image.decode_image(image_raw, num_channels)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image

def per_pixel_mean(dataset):
    """
    Compute the mean of each pixel over the entire dataset.

    """
    def collect_pixel():
        return

    return

def per_pixel_stddev(dataset):
    """
    Compute the stddev of each pixel over the entire dataset.

    """
    pass

def per_channel_mean_stddev(dataset):
    """
    Compute the mean & stddev of each channel for every image.
    """
    def channel_mean_stddev(decoded_image):
        means = tf.reduce_mean(decoded_image, axis=[0,1])
        stddev = tf.sqrt(tf.reduce_mean(tf.square(decoded_image-means), axis=[0,1]))
        return tf.stack([means, stddev])
    return dataset.map(lambda x: channel_mean_stddev(x))

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

a, b = load_images("D:/MURA-v1.1/train/*/*/*/*.png", 3 ,image_extension='png')
a= per_channel_mean_stddev(a)
a_iter = a.make_one_shot_iterator()
image = a_iter.get_next()
b_iter = b.make_one_shot_iterator()
label = b_iter.get_next()
with tf.Session() as sess:
    for i in range(10):
        print(sess.run([image, label]))