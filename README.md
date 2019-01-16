# Models-implementation
This is an implementation of CNN's architectures using TensorFlow. The main goal of this project is to support the creation of machine learning & deep learning Pipelines .

[lien](./setup.py)

## Description
The repository is composed of:
- Scripts for launching training and evaluation: It includes the basic tf.Session() and tf.train.MonitoredSession with different hooks and scaffold for distributed training.
- Folder Densenet: Regroup image preprocessing functions, script for a slim-like implementation of Densenet
- Folder utils: Convert data to Tfrecord format, analyse your data depending on its nature

##Installation

First, install Tensorflow (Or/And tensorflow-gpu) in order to perform computation in cpu-only (In GPU) 
Then, in order to use "slim" package, which is developped under : [lien](https://www.github.com/tensorflow/models/research/slim), 
git clone the above repository. Copy the "research" folder and perform both setup.py : 
- under research folder
- under slim folder (make sure you delete BUILD file before running setup.py)

As a last step, git clone Models-implementation repo : 
https://www.github.com/medtune/Models-implementation


## Usage example


## Release History
