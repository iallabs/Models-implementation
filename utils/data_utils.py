import tensorflow as tf
import PyPDF2
import os
import sys
import math
from .images.data_utils_image import compute_stats_fn, computes_stats, stats_to_tfexample, image_to_tfexample

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

def _get_filenames_and_multiclasses(dataset_dir):

    """Returns a list of filenames and inferred class names.
    Args:
    dataset_dir: A directory containing a set of subdirectories representing
    class names. Each subdirectory should contain PNG or JPG encoded images.

    Returns:
    A list of image file paths, relative to `dataset_dir` and the list of
    subdirectories, representing class names.
    """
    photo_filenames = []
    class_1_names = []
    class_2_names = []

    for root, _ , files in os.walk(dataset_dir):
        path = root.split(os.sep)
        for file in files:
            photo_filenames.append(os.path.join(root,file))
            class_1_names.append(path[-1].split("_")[-1])
            #TODO: The following is to encode the body part represented by the image
            class_2_names.append(path[2].split("_")[-1])
    return photo_filenames, class_1_names, class_2_names

def _get_train_valid(dataset_dir, multi=False):

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
    if multi:
        photos_train, class_1_train, class_2_train = _get_filenames_and_multiclasses(dataset_root_train)
        photos_valid, class_1_valid, class_2_valid = _get_filenames_and_multiclasses(dataset_root_valid)
        return photos_train, class_1_train, class_2_train,\
                photos_valid, class_1_valid, class_2_valid
    else:
        photos_train, class_train = _get_filenames_and_classes(dataset_root_train)
        photos_valid, class_valid = _get_filenames_and_classes(dataset_root_valid)
        return photos_train, class_train, photos_valid, class_valid


def _get_dataset_filename(dataset_dir, split_name, tfrecord_filename, stats=False):
    if stats:
        output_filename = os.path.join("stats",'%s_%s_stats.tfrecord' % (
                        tfrecord_filename, split_name))
    else:
        output_filename = '%s_%s.tfrecord' % (
                        tfrecord_filename, split_name)

    return os.path.join(dataset_dir, output_filename)


def _convert_dataset_bis(split_name, filenames, class_name, class_names_to_ids,
                         dataset_dir, tfrecord_filename, batch_size,
                         _NUM_SHARDS):
    """Converts the given filenames to a TFRecord dataset.
    (example: Ten different classes, each image has an unique class).
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
    lenght = len(filenames)
    output_filename = _get_dataset_filename(
                                dataset_dir, split_name, tfrecord_filename = tfrecord_filename,stats=False)
    output_filename_stats = _get_dataset_filename(
                                dataset_dir, split_name, tfrecord_filename = tfrecord_filename,stats=True)
    tfrecord_stats = tf.python_io.TFRecordWriter(output_filename_stats)
    
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer_1:
        for i in range(lenght):
            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            images_data.append(image_data)
            class_id = class_names_to_ids[class_name[i]]
            class_id_data.append(class_id)
            example_image = image_to_tfexample(image_data, class_id)
            tfrecord_writer_1.write(example_image.SerializeToString())
            if (i+1) % batch_size == 0 or i == lenght-1:
                with tf.Graph().as_default():
                    with tf.Session('') as sess:
                        gen_mean, gen_stddev, r_mean, r_stddev,\
                        g_mean, g_stddev, b_mean,\
                        b_stddev = computes_stats(sess, images_data)
                        for j in range(len(gen_mean)):
                            sys.stdout.write('\r>> Converting stats %d/%d' % (
                            i+1, lenght))
                            sys.stdout.flush()
                            #Py3: use encode("utf-8")
                            example = stats_to_tfexample(gen_mean[j],
                                                        gen_stddev[j], r_mean[j], r_stddev[j],
                                                        g_mean[j], g_stddev[j], b_mean[j],
                                                        b_stddev[j],class_name[j].encode(),
                                                        class_id_data[j])
                            tfrecord_stats.write(example.SerializeToString())
                images_data = []
                class_id_data = []   
    sys.stdout.write('\n')
    sys.stdout.flush()

def _convert_dataset_multi(split_name, filenames, class_first_name, class_snd_name, class_names_to_ids,
                         dataset_dir, tfrecord_filename, batch_size,
                         _NUM_SHARDS):
    """Converts the given filenames to a TFRecord dataset of multiple classes
       We have two types of labels, and we want to join them into an unique pair of keys
    Args:

        split_name: The name of the dataset, either 'train' or 'validation'.
        filenames: A list of absolute paths to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
        dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'eval']
    max_id = int(math.ceil(len(filenames) / float(batch_size)))
    output_filename = _get_dataset_filename(
                            dataset_dir, split_name, tfrecord_filename = tfrecord_filename,stats=False)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
        for i in range(len(filenames)):
            sys.stdout.write('\r>> Converting stats %d/%d' % (
                            i, len(filenames)))
            sys.stdout.flush()
            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            #TODO/The following line is Special to MURA dataset for defining 13 classes.
            class_id = class_names_to_ids[class_snd_name[i]+"_"+class_first_name[i]]
            example_image = image_to_tfexample(image_data, class_id)
            tfrecord_writer.write(example_image.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()