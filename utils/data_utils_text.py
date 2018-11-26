import tensorflow as tf
import os
import sys
import math
from utils.data_utils import int64_feature, bytes_feature, float_feature


def read_text_file(filenames, header=False):
    """
    Function to read text (.txt) files using the Dataset
    API
    Args:
    - filenames : list of one or more filenames to read from
    - header :  leave the header unchanged for value "True". Remove otherwise
    Returns:
    - dataset : a tf.Dataset object
    """
    if header:
        dataset = tf.data.TextLineDataset(filenames)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        # Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
        # and then concatenate their contents sequentially into a single "flat" dataset.
        # * Skip the first line (header row).
        # * Filter out lines beginning with "#" (comments).
        dataset = dataset.flat_map(
            lambda filename: (
                tf.data.TextLineDataset(filename)
                .skip(1)))
    pass

