import random

import tensorflow as tf

from utils.data_utils import _get_train_valid, _convert_dataset_bis



#====================================================DEFINE YOUR ARGUMENTS=======================================================================

flags = tf.app.flags

#State your dataset directory
flags.DEFINE_string('dataset_dir', None, 'String: Your dataset directory')

# The number of images in the validation set. You would have to know the total number of examples in advance. This is essentially your evaluation dataset.
flags.DEFINE_float('validation_size', 0.1, 'Float: The proportion of examples in the dataset to be used for validation')

# The number of shards to split the dataset into
flags.DEFINE_integer('num_shards', 1, 'Int: Number of shards to split the TFRecord files')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', None, 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

class_names_to_ids = {"negative": 0,
                        "positive":1}

def main():
    #==============================================================CHECKS==========================================================================

    #Check if there is a tfrecord_filename entered

    if not FLAGS.tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')



    #Check if there is a dataset directory entered

    if not FLAGS.dataset_dir:
        raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

    #==============================================================END OF CHECKS===================================================================

    #Get a list of photos filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
    photos_train, class_train, photos_valid, class_valid = _get_train_valid(FLAGS.dataset_dir)

    # First, convert the training and validation sets.
    """_convert_dataset_bis('train', photos_train, class_train, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, batch_size=500, _NUM_SHARDS=FLAGS.num_shards)"""
    _convert_dataset_bis('eval', photos_valid, class_valid, class_names_to_ids,
                     dataset_dir = FLAGS.dataset_dir, tfrecord_filename = FLAGS.tfrecord_filename, batch_size=200, _NUM_SHARDS=FLAGS.num_shards)

    print('\n Finished converting the %s dataset!' % (FLAGS.tfrecord_filename))

if __name__ == "__main__":
    main()