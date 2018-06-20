import random

import tensorflow as tf

from utils_csv import _dataset_exists, _get_infos, _convert_dataset

import pandas as pd

#====================================================DEFINE YOUR ARGUMENTS=======================================================================

flags = tf.app.flags

#State your dataset directory
flags.DEFINE_string('dataset_dir', None, 'String: Your dataset directory')

# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float('validation_size', 0.1, 'Float: The proportion of examples in the dataset to be used for validation')

# The number of shards to split the dataset into
flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files')


#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', None, 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

#TODO change this dict into names to ids
class_names_to_ids = {  'No Finding': 0,
                        'Atelectasis':1,
                        'Cardiomegaly':2,
                        'Effusion':3,
                        'Infiltration':4, 
                        'Mass':5,
                        'Nodule':6,
                        'Pneumonia':7,
                        'Pneumothorax':8,
                        'Consolidation':9,
                        'Edema':10,
                        'Emphysema':11,
                        'Fibrosis':12,
                        'Pleural_Thickening':13,
                        'Hernia':14}


def main():
    #==============================================================CHECKS==========================================================================

    #Check if there is a tfrecord_filename entered

    if not FLAGS.tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')



    #Check if there is a dataset directory entered

    if not FLAGS.dataset_dir:
        raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')



    #If the TFRecord files already exist in the directory, then exit without creating the files again

    if _dataset_exists(dataset_dir = FLAGS.dataset_dir, _NUM_SHARDS = FLAGS.num_shards, output_filename = FLAGS.tfrecord_filename):
        print('Dataset files already exist. Exiting without re-creating them.')
        return None

    #==============================================================END OF CHECKS===================================================================

    #Get a pandas dataframe containing each image path and text label
    grouped = _get_infos(FLAGS.dataset_dir)

    #Find the number of validation examples we need
    num_validation = int(FLAGS.validation_size * len(grouped))
    print(num_validation)


    # Divide the training datasets into train and test:
    training_filenames = pd.DataFrame.sample(grouped, frac=(1-FLAGS.validation_size))
    print(len(training_filenames))
    validation_filenames = grouped.loc[~grouped.index.isin(training_filenames.index), :]
    print(len(validation_filenames))



    # First, convert the training and validation sets.
    _convert_dataset('train', training_filenames, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)

    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, _NUM_SHARDS = FLAGS.num_shards)

    print('\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename))

if __name__ == "__main__":
    main()