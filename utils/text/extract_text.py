import tensorflow as tf
import os
import sys
import math
import pandas as pd
from bs4 import BeautifulSoup
#import nltk


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
    # Use `Dataset.flat_map()` to transform each file as a separate nested dataset,
    # and then concatenate their contents sequentially into a single "flat" dataset.
    # * Skip the first line (header row).
    # * Filter out lines beginning with "#" (comments).
    dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if header:
        dataset = dataset.flat_map(
            lambda filename: (
                tf.data.TextLineDataset(filename)))
    else:
        dataset = dataset.flat_map(
            lambda filename: (
                tf.data.TextLineDataset(filename)
                .skip(1)))
    return dataset

def parse_web_pages(url_list, structure_dict):
    """
    Having an url_list, we implement a function
    (with threading) in order to parse web pages of the same root
    url. We also provide a structure dict with keys as the page's title
    (ex: HOME/ About) and values are also dicts of (keys= HTML tags, 
    values=class/id name). 
    """
    pass

def extract_from_url(url, structure):
    """
    Given an url and the structure of the HTML to parse,
    we use BeautifulSoup to extract useful text from web page 
    """
    pass

def clean_html(html_text, structure):
    """
    Utility function to use inside "extract_from_url"
    """
    pass

def extract_from_pdf(filename_pattern):
    """
    Using PDF2, we parse a PDF and target divisions where
    useful text is located 
    """
    pass

def write_into_txt(inputs, sentences_filename, labels_filename):
    """
    Given text inputs, we want to write in a structured manner:
    1 line == 1 training example
    """
    pass

def write_into_csv(inputs, filename, labels_filename):
    """
    Given text inputs, we want to encode in a structred csv manner
    First line : the header of the file. The following is an example
    of comlumn names (sentence, label, ...)
    """
    pass