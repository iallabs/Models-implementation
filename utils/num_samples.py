import tensorflow as tf
import os
dataset_dir = os.path.join("D:","Oxford_pets")
phase_name = "val"
file_pattern_for_counting = "pet_faces"
num_samples = 0
file_pattern_for_counting = file_pattern_for_counting + '_' + phase_name
tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if file.startswith(file_pattern_for_counting)]
for tfrecord_file in tfrecords_to_count:
    for record in tf.python_io.tf_record_iterator(tfrecord_file):
        num_samples += 1
print(num_samples)