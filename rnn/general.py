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
    cells : list of rnn classes(basic/lstm/gru)
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
    last_result = tf.gather(output, int(output.get_shape()[0]) - 1)
    #Add a classification layer (dense layer)
    return last_result

def classify_and_loss(last_result, labels):
    """
    Logit : final result of the last rnn layer, after applying
    readyfor_sequence_classification. For classification, we care about 
    the output activation at the last timestep, which is justoutputs[-1]
    Classification : output a result after the computation of the sequence
    """
    num_class = labels.get_shape()[2].value
    logits = tf.layers.dense(last_result, num_class)
    predictions = tf.nn.softmax(logits)
    loss = tf.losses.softmax_cross_entropy(labels, logits)
    return predictions, loss

def labelling_and_loss(last_result, labels):
    """
    Labelling : get a classification for each timestep (each input)
    """
    # We share the weights for the softmax layer across all timesteps
    # by flattening the first two dimensions of the output tensor
    #We have a prediction and a label for each time step.
    # We compute the cross entropy for every time step and sequence in the batch
    num_class = labels.get_shape()[2].value
    logits = tf.layers.dense(last_result, num_class)
    predictions = tf.nn.softmax(logits)
    flat_labels = tf.reshape(labels, [-1] + labels.shape.as_list()[2:])
    flat_logits = tf.reshape(logits, [-1] + logits.shape.as_list()[2:])
    loss = tf.losses.softmax_cross_entropy(flat_labels, flat_logits)
    loss = tf.reduce_mean(loss)
    return flat_logits, loss