import tensorflow as tf
import PyPDF2
import os
import sys
import math

#SOURCE: https://github.com/tensorflow/models/blob/master/research/slim/datasets/download_and_convert_flowers.py

slim = tf.contrib.slim

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

def computes_stats(sess, images_data, batch_size):
    images = tf.placeholder(dtype=tf.string, shape=[batch_size])
    results = tf.map_fn(lambda x: compute_stats_fn(x), images, dtype=tf.float32)
    alpha = sess.run(results, feed_dict={images:images_data})
    GEN_mean, GEN_stddev, R_mean,\
    R_stddev, G_mean, G_stddev, B_mean,\
    B_stddev = (alpha[:,s] for s in range(8))

    return GEN_mean, GEN_stddev, R_mean,\
            R_stddev, G_mean, G_stddev, B_mean,\
            B_stddev

def parse_pdf(pdf_filename):
    """Function to parse PDF file
    It'll return a list of images extracted from
    the PDF, the number of pages(?) and also
    the filename. pdf_filename would be the 
    """
    pdf_file = open(pdf_filename)
    read_pdf = PyPDF2.PdfFileReader(pdf_file)
    for i in range(read_pdf.getNumPages()):
        page = read_pdf.getPage(i)
        # The tag /Contents is required. If not content = None
        content = page.getContents()
        pass
    
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

def float_feature(value):
    if not isinstance(value, list):
        values=[value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

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
        "image/class/label": int64_feature(label)
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

class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""
    def __init__(self):
        # Initializes function that decodes RGB JPEG data.
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_image(self._decode_jpeg_data, channels=3)

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
    photo_filenames = []
    
    class_names = []

    for root, _ , files in os.walk(dataset_dir):
        path = root.split(os.sep)
        for file in files:
            photo_filenames.append(os.path.join(root,file))
            class_names.append(path[-1].split("_")[-1])

    return photo_filenames, class_names

def _get_train_valid(dataset_dir):

    """Returns a list of filenames and inferred class names.
    This function needs a defined train and validayion folders
    Args:
    dataset_dir: A directory containing a set of subdirectories representing
    class names. Each subdirectory should contain PNG or JPG encoded images.
    Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
    """
    dataset_root_train = os.path.join(dataset_dir, "train")
    dataset_root_valid = os.path.join(dataset_dir, "valid")
    photos_train, class_train = _get_filenames_and_classes(dataset_root_train)
    photos_valid, class_valid = _get_filenames_and_classes(dataset_root_valid)

    return photos_train, class_train, photos_valid, class_valid

def _get_dataset_filename(dataset_dir, split_name, tfrecord_filename, stats=False):
    if stats:
        output_filename = '%s_%s_stats.tfrecord' % (
                        tfrecord_filename, split_name)
    else:
        output_filename = '%s_%s.tfrecord' % (
                        tfrecord_filename, split_name)

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

    assert split_name in ['train', 'eval']
    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                            dataset_dir, split_name, tfrecord_filename = tfrecord_filename)
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i+1, len(filenames), shard_id))
                    sys.stdout.flush()
                    # Read the filename:
                    image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                    class_name = os.path.basename(os.path.dirname(filenames[i]))
                    class_id = class_names_to_ids[class_name]
                    example = image_to_tfexample(image_data,class_id)
                    tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

def _convert_dataset_bis(split_name, filenames, class_name, class_names_to_ids,
                         dataset_dir, tfrecord_filename, batch_size,
                         _NUM_SHARDS):
    """Converts the given filenames to a TFRecord dataset.
    Args:

        split_name: The name of the dataset, either 'train' or 'validation'.
        filenames: A list of absolute paths to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
        dataset_dir: The directory where the converted datasets are stored.

    """
    images_data = []
    class_id_data = []
    assert split_name in ['train', 'eval']
    max_id = int(math.ceil(len(filenames) / float(batch_size)))
    output_filename = _get_dataset_filename(
                                dataset_dir, split_name, tfrecord_filename = tfrecord_filename,stats=False)
    output_filename_stats = _get_dataset_filename(
                                dataset_dir, split_name, tfrecord_filename = tfrecord_filename,stats=True)
    
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer_1:
        for i in range(len(filenames)):
            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            images_data.append(image_data)
            class_id = class_names_to_ids[class_name[i]]
            class_id_data.append(class_id)
            example_image = image_to_tfexample(image_data, class_id)
            tfrecord_writer_1.write(example_image.SerializeToString())
    with tf.Graph().as_default():
        with tf.Session('') as sess:
            with tf.python_io.TFRecordWriter(output_filename_stats) as tfrecord_writer:
                for i in range(max_id):
                    start_ndx = i * batch_size
                    end_ndx = min((i+1) * batch_size, len(filenames))
                    try:
                        gen_mean, gen_stddev, r_mean, r_stddev,\
                        g_mean, g_stddev, b_mean,\
                        b_stddev = computes_stats(sess, images_data[start_ndx:end_ndx], end_ndx-start_ndx)
                        for j in range(len(gen_mean)):
                            sys.stdout.write('\r>> Converting stats %d/%d shard %d' % (
                            j+start_ndx, len(filenames), i))
                            sys.stdout.flush()
                            #Py3: use encode("utf-8")
                            example = stats_to_tfexample(gen_mean[j],
                                                        gen_stddev[j], r_mean[j], r_stddev[j],
                                                        g_mean[j], g_stddev[j], b_mean[j],
                                                        b_stddev[j],class_name[start_ndx+j].encode(),
                                                        class_id_data[start_ndx+j])
                            tfrecord_writer.write(example.SerializeToString())
                    except:
                        print("batch of image is corrupted")
    sys.stdout.write('\n')
    sys.stdout.flush()

