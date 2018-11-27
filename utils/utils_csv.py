import tensorflow as tf
import pandas as pd
import os
import sys
import math
from object_detection.utils import dataset_util
#SOURCE: https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_flowers.py

slim = tf.contrib.slim
#############################################
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

##########################################

def image_to_tfexample(image_data,filename, image_format, height, width,
                            class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/filename': bytes_feature(filename),
        'image/format': bytes_feature(image_format),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/class/id': int64_feature(class_id),
    }))

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""
    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]
    
    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                        feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image  


def _get_dataset_filename(dataset_dir, split_name, shard_id, tfrecord_filename, _NUM_SHARDS):

    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
                        tfrecord_filename, split_name, shard_id, _NUM_SHARDS)

    return os.path.join(dataset_dir, output_filename)

def _get_infos(dataset_dir, csv_name):
    """Data_Entry_csv is the csv filename discribing"""
    path_info = os.path.join(dataset_dir, csv_name)
    grouped = pd.read_csv(path_info)
    
    return grouped

##########################################################

def _convert_dataset(split_name, grouped, class_names_to_ids, dataset_dir, tfrecord_filename, _NUM_SHARDS):
    """Converts the given filenames to a TFRecord dataset.
    Args:
        split_name: The name of the dataset, either 'train' or 'validation'.
        dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'eval']
    num_per_shard = int(math.ceil(len(grouped) / float(_NUM_SHARDS)))
    path_img = os.path.join(dataset_dir, 'images')#: To use if images are located in one folder named images
    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                                dataset_dir, split_name, shard_id, tfrecord_filename = tfrecord_filename, _NUM_SHARDS = _NUM_SHARDS)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(grouped))

                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i+1, len(grouped), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        row = grouped.iloc[i]
                        image_data = tf.gfile.FastGFile(os.path.join(path_img, row[0]), 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)
                        #Special to ChestX Dataset: we only focus on the first anomaly(see Data_Entry_csv)
                        if '|' in row['Finding Labels']:
                            class_name = row['Finding Labels'].split('|')
                            class_id = [class_names_to_ids[s] for s in class_name]
                        else: 
                            class_name = row['Finding Labels']
                            class_id = class_names_to_ids[class_name]
                        example = image_to_tfexample(image_data, row[0].encode(), 'png'.encode(),
                                                    height, width, class_id)
        
                        tfrecord_writer.write(example.SerializeToString())
                        tfrecord_writer.flush()
    sys.stdout.write('\n')
    sys.stdout.flush()

#############################################################

def _dataset_exists(dataset_dir, _NUM_SHARDS, output_filename):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            tfrecord_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id, output_filename, _NUM_SHARDS)
            if not tf.gfile.Exists(tfrecord_filename):
                return False

    return True