import tensorflow as tf
import os

def load_images(filenames_pattern, image_extension='jpg'):
    """
    we want to load data into Datasets in order to perform
    different map-fn fucntions preprocessing

    Args:
    filenames_list : A string representing path pattern of each image
    
    Return:
    tf.data.Dataset containing raw image data, decoded and flattened
    """
    dataset = tf.data.Dataset.list_files(filenames_pattern, shuffle=False)
    label_dataset = dataset.map(lambda x: extract_label(x, label_pos=-2), num_parallel_calls=100)
    image_dataset = dataset.map(lambda x: extract_image(x, image_extension), num_parallel_calls=100)
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

def extract_image(filename, image_extension='jpg'):
    image_raw = tf.read_file(filename)
    if image_extension == 'jpg':
        image = tf.image.decode_jpeg(image_raw)
    else:
        image = tf.image.decode_png(image_raw)
    return image

def per_pixel_mean(dataset):
    """
    Compute the mean of each pixel over the entire dataset.

    """
    pass

def per_pixel_stddev(dataset):
    """
    Compute the stddev of each pixel over the entire dataset.

    """
    pass

def per_channel_mean(dataset):
    """
    Compute the mean of each channelf for every image.

    """
    pass

def per_channel_stddev(dataset):
    """
    Compute the stddev of each channel for every image.

    """
    pass

def encode_stats(alpha):
    """
    Utility function that encodes Mean per pixel
    depending on height*width of image
    """
    pass

a, b = load_images("D:/MURA-v1.1/train/*/*/*/*.png", 'png')
a_iter = a.make_one_shot_iterator()
image = a_iter.get_next()
b_iter = b.make_one_shot_iterator()
label = b_iter.get_next()
with tf.Session() as sess:
    print(sess.run([image, label]))