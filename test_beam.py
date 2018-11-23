import os

import apache_beam as beam
import tensorflow as tf
from apache_beam.options.pipeline_options import PipelineOptions

directory = "D:/MURA-v1.1"
file_pattern = os.path.join(directory, "mura_*.tfrecord")

p = beam.Pipeline(options=PipelineOptions())

#Read from tfrecord file:

example = (p | "ReadTF" >> beam.io.tfrecordio.ReadFromTFRecord(file_pattern)
             | "DecodeImage" >> )