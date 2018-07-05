import tensorflow as tf
import os
import sys
import math

#SOURCE: https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_flowers.py

slim = tf.contrib.slim

def int64_feature(value):
    """ Returns a TF-feature of int64
        Args: value: scalar or list of values
        return: TF-Feature"""
    if not isinstance(value, (tuple, list)):
        values = [value]

    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def bytes_feature(value):
    """Return a TF-feature of bytes"""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def image_to_tfexample(image_data, image_format, height, width, class_id):
  return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
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

def _get_filenames_and_classes(dataset_dir):

    """Returns a list of filenames and inferred class names.
    Args:
    dataset_dir: A directory containing a set of subdirectories representing
    class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
    """
    # print 'DATASET DIR:', dataset_dir
    # print 'subdir:', [name for name in os.listdir(dataset_dir)]
    # dataset_main_folder_list = []
    # for name in os.listdir(dataset_dir):
    # 	if os.path.isdir(name):
    # 		dataset_main_folder_list.append(name)

    dataset_main_folder_list = [name for name in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,name))]
    print(dataset_main_folder_list)
    dataset_root = os.path.join(dataset_dir, dataset_main_folder_list[1])

    directories = []

    class_names = []

    for filename in os.listdir(dataset_root):
        path = os.path.join(dataset_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []

    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id, tfrecord_filename, _NUM_SHARDS):

    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
                        tfrecord_filename, split_name, shard_id, _NUM_SHARDS)

    return os.path.join(dataset_dir, output_filename)


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir, tfrecord_filename, _NUM_SHARDS):
    """Converts the given filenames to a TFRecord dataset.
    Args:

        split_name: The name of the dataset, either 'train' or 'validation'.
        filenames: A list of absolute paths to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
        dataset_dir: The directory where the converted datasets are stored.

    """

    assert split_name in ['train', 'validation']
    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                                dataset_dir, split_name, shard_id, tfrecord_filename = tfrecord_filename, _NUM_SHARDS = _NUM_SHARDS)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                        i+1, len(filenames), shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(sess, image_data)
                        class_name = os.path.basename(os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]
                        print(class_name)
                        print(filenames[i])
                        example = image_to_tfexample(image_data, 'jpg'.encode(), height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _dataset_exists(dataset_dir, _NUM_SHARDS, output_filename):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            tfrecord_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id, output_filename, _NUM_SHARDS)
            if not tf.gfile.Exists(tfrecord_filename):
                return False

    return True