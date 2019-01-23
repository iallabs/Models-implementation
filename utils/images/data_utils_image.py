import tensorflow as tf
import os
import sys
import math

def int64_feature(value):
    """ Returns a TF-feature of int64
        Args: value: scalar or list of values
        return: TF-Feature"""
    if not isinstance(value, (tuple, list)):
        values = [value]
    else:
        values = value
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(value):
    """Return a TF-feature of bytes"""
    if not isinstance(value, (tuple, list)):
        values = [value]
    else:
        values = value
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def float_feature(value):
    if not isinstance(value, list):
        values=[value]
    else:
        values = value
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def compute_stats_fn(image_data):
    
    image_u = tf.image.decode_image(image_data, channels=3)  
    image_f = tf.image.convert_image_dtype(image_u, dtype=tf.float32)
    gen_mean, gen_stddev = tf.nn.moments(image_f, axes=[0,1,2])
    c_image = tf.split(axis=2, num_or_size_splits=3, value=image_f)
    Rc_image, Gc_image, Bc_image = c_image[0], c_image[1], c_image[2]
    r_mean, r_var = tf.nn.moments(Rc_image, axes=[0,1])
    g_mean, g_var = tf.nn.moments(Gc_image, axes=[0,1])
    b_mean, b_var = tf.nn.moments(Bc_image, axes=[0,1])
    r_stddev = tf.sqrt(r_var)
    g_stddev = tf.sqrt(g_var)
    b_stddev = tf.sqrt(b_var)
    result = tf.stack([gen_mean, gen_stddev, tf.squeeze(r_mean), tf.squeeze(r_stddev),\
                        tf.squeeze(g_mean), tf.squeeze(g_stddev),\
                        tf.squeeze(b_mean), tf.squeeze(b_stddev)])
    return result

def computes_stats(sess, images_data):
    images = tf.placeholder(dtype=tf.string, shape=[None])
    results = tf.map_fn(lambda x: compute_stats_fn(x), images, dtype=tf.float32,
                        parallel_iterations=4)
    alpha = sess.run(results, feed_dict={images:images_data})
    GEN_mean, GEN_stddev, R_mean,\
    R_stddev, G_mean, G_stddev, B_mean,\
    B_stddev = (alpha[:,s] for s in range(8))

    return GEN_mean, GEN_stddev, R_mean,\
            R_stddev, G_mean, G_stddev, B_mean,\
            B_stddev

def stats_to_tfexample(gen_mean,gen_stddev,
                        r_mean, r_stddev, g_mean, g_stddev,
                        b_mean,b_stddev, class_name, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/class/str':bytes_feature(class_name),
        'image/class/label': int64_feature(class_id),
        'image/stats/gen_mean': float_feature(gen_mean),
        'image/stats/gen_stddev': float_feature(gen_stddev),
        'image/stats/r_mean': float_feature(r_mean),
        'image/stats/r_stddev': float_feature(r_stddev),
        'image/stats/g_mean': float_feature(g_mean),
        'image/stats/g_stddev': float_feature(g_stddev),
        'image/stats/b_mean': float_feature(b_mean),
        'image/stats/b_stddev': float_feature(b_stddev)
    }))

def image_to_tfexample(image_data, label):
    return tf.train.Example(features=tf.train.Features(feature={
        "image/encoded": bytes_feature(image_data),
        "image/class/id": int64_feature(label)
    }))

def multi_task_tfexample(image_data, label_1, label_2):
    """
    This is a utility function to encode a dataset in order
    to perform a multi task training and evaluation
    """
    return tf.train.Example(features=tf.train.Features(feature={
        "image/encoded": bytes_feature(image_data),
        "image/class/id_1": int64_feature(label_1),
        "image/class/id_2": int64_feature(label_2),
    }))

#TODO: Waiting for URL construction to decide how to split the URL
# For now, we're going to write each pdf data and it's corresponding information
#for each example
def pdf_to_tfexample(pdf_data, repo, typo, spec, origin, url):
    #We'll assume that we're working with python2.7 version
    # in order to satisfy the Apache Beam dependencie
    repo, typo, spec,\
    origin, url = repo.encode(), typo.encode(),\
                    spec.encode(), origin.encode(),\
                    url.encode()

    return tf.train.Example(features=tf.train.Features(feature={
        "image/encoded": bytes_feature(pdf_data),
        "image/class/repo": bytes_feature(repo),
        "image/class/typo": bytes_feature(typo),
        "image/class/spec": bytes_feature(spec),
        "image/class/origin": bytes_feature(origin),
        "image/filename" : bytes_feature(url)
    }))