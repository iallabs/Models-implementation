import os


import numpy as np
import pandas as pd
import tensorflow as tf

import tqdm



def compute_statistics(paths, verbose=False):
    """Computes the mean and standard deviation of NIH Chest Xray intensities
    Args:
        s: Paths to all images in NIH Chest Xray data
        verbose: Print progress and final results

    Returns:
        mean, std: Statistics of intensities
    """

    sess = tf.Session()



    batch_size = 100
    dataset = tf.data.Dataset.from_tensor_slices({'path': paths})
    dataset = dataset.map(
        lambda path: decode_image(read_file(path)),
        num_parallel_calls=32
    )

    dataset = dataset.batch(batch_size).prefetch(16)

    iterator = dataset.make_one_shot_iterator()
    batch = iterator.get_next()
    images = batch['image']



    mean_metric = tf.metrics.mean
    moments, update_ops = list(
        zip(mean_metric(images), mean_metric(images**2))

    )



    sess.run(tf.local_variables_initializer())
    batches = range(len(paths) // batch_size)
    if verbose:

        batches = tqdm.tqdm(batches, unit='batch')
    for _ in batches:
        moment1, moment2 = sess.run(update_ops)
        mean = moment1
        std = np.sqrt(moment2 - moment1**2)
        if verbose:
            batches.set_description('mean: %.3f, std: %.3f' % (mean, std))



    if verbose:
        print('Mean: ', mean)  
        print('Std: ', std)    
    return mean, std





def main():
    main_dir = ""
    raw_data = pd.read_csv(os.path.join(main_dir, 'train_image_path.csv'))
    paths = raw_data[0]
    compute_statistics(paths.values, verbose=True)