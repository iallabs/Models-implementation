import random
import os
import tensorflow as tf
import yaml
from utils.data_utils import _get_train_valid, _convert_dataset_bis



#====================================================DEFINE YOUR ARGUMENTS=======================================================================
stream = open(os.path.join(os.getcwd(), "yaml","config_tfrec.yaml"))
data = yaml.load(stream)
print(data)
dataset_dir = data["dataset_dir"]
tfrecord_filename = data["tfrecord_filename"]
validation_size = data["validation_size"]
num_shards = data["num_shards"]
class_names_to_ids = data["class_names_to_ids"]


def main():
    #==============================================================CHECKS==========================================================================

    #Check if there is a tfrecord_filename entered

    if not tfrecord_filename:
        raise ValueError('tfrecord_filename is empty. Please state a tfrecord_filename argument.')



    #Check if there is a dataset directory entered

    if not dataset_dir:
        raise ValueError('dataset_dir is empty. Please state a dataset_dir argument.')

    #==============================================================END OF CHECKS===================================================================

    #Get a list of photos filenames like ['123.jpg', '456.jpg'...] and a list of sorted class names from parsing the subdirectories.
    photos_train, class_train, photos_valid, class_valid = _get_train_valid(dataset_dir)
    
    # First, convert the training and validation sets.
    _convert_dataset_bis('train', photos_train, class_train, class_names_to_ids,
                     dataset_dir = dataset_dir, tfrecord_filename = tfrecord_filename, batch_size=500, _NUM_SHARDS = num_shards)
    _convert_dataset_bis('eval', photos_valid, class_valid, class_names_to_ids,
                     dataset_dir = dataset_dir, tfrecord_filename = tfrecord_filename, batch_size=200, _NUM_SHARDS = num_shards)

    print('\n Finished converting the %s dataset!' % (tfrecord_filename))

if __name__ == "__main__":
    main()