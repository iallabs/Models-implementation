import tensorflow as tf
import os
import sys
import math
#import nltk


def read_text_file_pattern(file_pattern, header=False):
    pass

def split_into_tokens(dataset, delimiter=""):
    """
    Depending on the type of the tokens (characters, sentences, documents)
    and the delimiter, we perform a string_split operation on the dataset.
    Thus, we define an "extract_func" the will be called during dataset.map
    """
    pass

def extract_sentence(tensor, delimiter=""):
    """
    Utility function to be used during dataset.map operation
    tensor represent a sequence of sentences
    """
    pass

def extract_character(tensor, delimiter=""):
    """
    Utility function to be used during dataset.map operation
    tensor represent a unique sentence
    """
    pass

def vocab_tables(source_file, tags_file):
    """
    Using lookup_table operation, we build our vocabulary table
    It depend of the tokens we are using (character/sentence)
    """
    pass

def sequence_to_list_ids(sequence, vocab):
    """
    Having decided on a vocabulary, we convert each sequence to the
    corresponding list of ids
    """
    pass