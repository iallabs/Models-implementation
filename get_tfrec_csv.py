import random

import tensorflow as tf

from utils.utils_csv import _dataset_exists, _get_infos, _convert_dataset_bis

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
class_names_to_ids = {'negative': 0,
                    'positive':1}
"""class_names_to_ids = {
                'No Finding':0, 
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
                'Hernia':14
                }"""

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
    """csv_name = "train_image_paths.csv"
    #Get a pandas dataframe containing each image path and text label
    training_filenames = _get_infos(FLAGS.dataset_dir, csv_name)
    training_frame = pd.DataFrame.sample(training_filenames, frac=1, random_state=1)
    print(len(training_frame))

    csv_eval_name = "valid_image_paths.csv"
    #Get a pandas dataframe containing each image path and text label
    valid_filenames = _get_infos(FLAGS.dataset_dir, csv_eval_name)
    valid_frame = pd.DataFrame(valid_filenames)
    print(len(valid_frame))"""
    grouped=_get_infos(FLAGS.dataset_dir,"Data_Entry_2017.csv")
    # Divide the training datasets into train and test:(For ChestX like datasets)
    
    training_filenames = pd.DataFrame.sample(grouped, frac=(1-FLAGS.validation_size))
    training_frame = pd.DataFrame.sample(training_filenames, frac=1,random_state=3)
    print(len(training_frame))
    validation_filenames = grouped.loc[~grouped.index.isin(training_filenames.index), :]
    valid_frame = pd.DataFrame.sample(validation_filenames, frac=1,random_state=3)
    print(len(valid_frame))



    # First, convert the training and validation sets.
    _convert_dataset_bis('train', photos_train, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, batch_size=500, _NUM_SHARDS=FLAGS.num_shards)
    _convert_dataset_bis('eval', photos_valid, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename,batch_size=100, _NUM_SHARDS=FLAGS.num_shards)

    print('\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename))

if __name__ == "__main__":
    main()