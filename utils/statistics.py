import os


import numpy as np
import pandas as pd
import tensorflow as tf

import cv2


def compute_statistics(paths, verbose=False):
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
        image = tf.image.resize_bilinear(tf.expand_dims(image,0), (224,224))
        return image
    dataset = dataset.map(
        parse_fn,
        num_parallel_calls=32
    )
    dataset = dataset.batch(batch_size).prefetch(16)

    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    images = batch[0]



    mean_metric = tf.metrics.mean
    moments, update_ops = list(
        zip(mean_metric(images), mean_metric(images**2))

    )



    sess.run(tf.local_variables_initializer())
    txt_file = open("statistics-mura.txt","w")
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
    return mean, stddev





def main():
    main_dir = "D:/MURA-v1.1"
    raw_data = pd.read_csv(os.path.join(main_dir, 'train_image_paths.csv'))
    paths = raw_data[0:].values
    paths = [os.path.join(main_dir,val[0]) for val in paths]
    compute_statistics(paths)

if __name__=='__main__':
    main()