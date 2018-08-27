import os


import numpy as np
import pandas as pd
import tensorflow as tf

import cv2

def stddev(moment1, moment2):
    stddev = np.sqrt(moment2 - moment1**2)
    return stddev

def compute_statistics(paths, verbose=False):
    """Computes the mean and standard deviation of NIH Chest Xray intensities
    Args:
        s: Paths to all images in NIH Chest Xray data
        verbose: Print progress and final results

    Returns:
        mean, std: Statistics of intensities
    """
    sess = tf.Session()

    batch_size = 1000
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    
    def parse_fn(filename):
        image_str = tf.read_file(filename)
        image = tf.image.decode_image(image_str, channels=3)
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.image.resize_bilinear(tf.expand_dims(image,0), (224,224))[0]
        return image
    dataset = dataset.map(
        parse_fn,
        num_parallel_calls=32
    )
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()

    mean_metric = tf.metrics.mean
    moments, update_ops = list(
        zip(mean_metric(batch), mean_metric(batch**2))
    )
    sess.run(tf.local_variables_initializer())
    txt_file = open("statistics-chest.txt","w")
    gen_mean, gen_stddev = 0.0, 0.0
    batches = len(paths) // batch_size
    for i in range(batches):
        moment1, moment2 = sess.run(update_ops)
        mean = moment1
        gen_mean += mean
        stddev = np.sqrt(moment2 - moment1**2)
        gen_stddev += stddev 
        txt_file.write("batch-%d-mean-%.3f-stddev-%.3f\n"%(i, mean, stddev))
    txt_file.write("\ntotal mean: %.3f, total stddev: %.3f\n"%(gen_mean/batches, gen_stddev/batches))
    txt_file.close()


def compute_statistics_channels(paths, verbose=False):
    """Computes the mean and standard deviation of NIH Chest Xray intensities
    Args:
        s: Paths to all images in NIH Chest Xray data
        verbose: Print progress and final results

    Returns:
        mean, std: Statistics of intensities
    """
    sess = tf.Session()
    batch_size = 16
    dataset = tf.data.Dataset.from_tensor_slices(paths)
    
    def parse_fn(filename):
        image_str = tf.read_file(filename)
        image = tf.image.decode_image(image_str, channels=3)
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.split(axis=2, num_or_size_splits=3, value=image)
        image = tf.image.resize_bilinear(image, (224,224))
        
        return image
    dataset = dataset.map(
        parse_fn,
    )
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    R_channel = batch[0]
    G_channel = batch[1]
    B_channel = batch[2]
    mean_metric = tf.metrics.mean
    moments, update_ops = list(
        zip(mean_metric(R_channel), mean_metric(R_channel**2),
            mean_metric(G_channel), mean_metric(G_channel**2),
            mean_metric(B_channel), mean_metric(B_channel**2)))

    sess.run(tf.local_variables_initializer())
    txt_file = open("statistics-channels-chest.txt","w")
    gen_R_mean, gen_R_stddev,gen_G_mean, gen_G_stddev,gen_B_mean, gen_B_stddev = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    batches = len(paths) // batch_size
    for i in range(batches):
        R_moment1, R_moment2, G_moment1, G_moment2, B_moment1, B_moment2 = sess.run(update_ops)
        gen_R_mean += R_moment1
        gen_G_mean += G_moment1
        gen_B_mean += B_moment1
        R_stddev, G_stddev, B_stddev = stddev(R_moment1, R_moment2),\
                                       stddev(G_moment1, G_moment2),\
                                       stddev(B_moment1, B_moment2)
        gen_R_stddev += R_stddev
        gen_G_stddev += G_stddev
        gen_B_stddev += B_stddev
        txt_file.write("batch-%d,R_mean-%.3f,R_stddev-%.3f,\
                        G_mean-%.3f,G_stddev-%.3f,\
                        B_mean-%.3f,B_stddev-%.3f\n"%(i, R_moment1, R_stddev,
                                                        G_moment1, G_stddev,
                                                        B_moment1, B_stddev))
        txt_file.flush()                                                
    txt_file.write("\n total R_mean: %.3f, total R_stddev: %.3f,\
                    total G_mean: %.3f, total G_stddev: %.3f,\
                    total B_mean: %.3f, total B_stddev: %.3f,\
                    \n"%(gen_R_mean/batches, gen_R_stddev/batches,
                         gen_G_mean/batches, gen_G_stddev/batches,
                         gen_B_mean/batches, gen_B_stddev/batches,))
    txt_file.close()

def main():
    main_dir = "D:/ChestXray-14"
    image_dir=os.path.join(main_dir, "images")
    raw_data = pd.read_csv(os.path.join(main_dir, 'Data_Entry_2017.csv'))
    paths = raw_data[0:].values
    paths = [os.path.join(image_dir,val[0]) for val in paths]
    compute_statistics_channels(paths)

if __name__=='__main__':
    main()