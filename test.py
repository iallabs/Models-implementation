import tensorflow as tf


record_iterator = tf.python_io.tf_record_iterator(path='D:/chest/chest_eval_00000-of-00001.tfrecord')
for string_record in record_iterator:
    example = tf.train.Example()
    example.ParseFromString(string_record)
    i = example.features.feature['image/class/id'].int64_list.value
    print(i)