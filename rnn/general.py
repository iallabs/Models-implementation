import tensorflow as tf
import numpy as np
import pandas as pd


#We create functions to efficiently contruct a model_fn
#function. We include dropout for training phase
#
def build_rnn_cell(type, num_units, dropout):
    """
    Construct a basic RNN cell. It is configured to be passed to a
    multi rnn cell wrapper. 
    """
    cell = tf.nn.rnn_cell.BasicRNNCell(num_units)
    if dropout:
        result = tf.nn.rnn_cell.DropoutWrapper(cell,
                                     output_keep_prob=1-dropout)
    return result

def build_lstm_cell(num_units, dropout):
    """
    Construct a basic RNN cell. It is configured to be passed to a
    multi rnn cell wrapper.
    """
    cell = tf.nn.rnn_cell.LSTMCell(num_units)
    if dropout:
        result = tf.nn.rnn_cell.DropoutWrapper(cell,
                                     output_keep_prob=1-dropout)
    return result

def build_gru_cell(num_units, dropout):
    """
    Construct a GRU RNN cell. It is configured to be passed to a
    multi rnn cell wrapper.
    """
    cell = tf.nn.rnn_cell.GRUCell(num_units)
    if dropout:
        result = tf.nn.rnn_cell.DropoutWrapper(cell,
                                     output_keep_prob=1-dropout)
    return result

def build_multi_rnn(cells):
    """
    Multi RNN cell wraps a sequence of cells into one cell
    """
    return tf.nn.rnn_cell.MultiRNNCell(cells)

#Add operation to the graph and simulate the recurrent network over
#the time steps of the input
def dynamic_rnn(cell, input_data):
    """
    cell : rnn cell (lstm/ gru/ basic or multicell)
    input_data. Tensor of type float32 and size 
            batch_size * time_steps * features
    Return:
        - output: activations computed from rnn
        - state : hidden state of rnn
    """
    output, state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    return output, state

def readyfor_sequence_classification(output):
    """
    output : activations computed from rnn.
    In this case, we are not interested in hidden state
    (last previous 'memory' of rnn)
    """
    #Transpose to make the time axis the first dimension:
    output = tf.transpose([1,0,2])
    #tf.gather to select the last frame (output[-1])
    logit = tf.gather(output, int(output.get_shape()[0]) - 1)
    #Add a classification layer (dense layer)
    return logit

