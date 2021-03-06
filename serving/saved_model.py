import tensorflow as tf
slim = tf.contrib.slim

import research.slim.nets.mobilenet.mobilenet_v2 as mobilenet_v2
from research.slim.preprocessing import inception_preprocessing
import DenseNet.preprocessing.densenet_pre as dp
import DenseNet.nets.densenet as densenet
import research.slim.nets.inception_resnet_v2 as inception


import os
import numpy as np


tf.app.flags.DEFINE_integer('model_version', 2, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

checkpoint_dir = os.getcwd()
checkpoint_file = "D:/ckpt-mura.05.08.2018/model-115001"

image_size = 224
#Images
#Labels MURA
labels_to_name = {0:'negative', 
                1:'positive'
                }

#Labels of chest-X-Ray:
labels_to_name = {
                0:'Atelectasis',
                1:'Cardiomegaly', 
                2:'Effusion',
                3:'Infiltration',
                4:'Mass',
                5:'Nodule',
                6:'Pneumonia',
                7:'Pneumothorax',
                8:'Consolidation',
                9:'Edema',
                10:'Emphysema',
                11:'Fibrosis',
                12:'Pleural_Thickening',
                13:'Hernia'
                }
tf.logging.set_verbosity(tf.logging.INFO)

serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
feature_configs = {'x':tf.FixedLenFeature((),tf.string)}
tf_example = tf.parse_single_example(serialized_tf_example, feature_configs)
image = tf.image.decode_image(tf_example['x'], channels=3)
image = tf.image.convert_image_dtype(image, tf.float32)
image.set_shape([None,None,3])
image_a = dp.preprocess_image(image, 224,224, is_training=False)
image_bis = tf.expand_dims(image_a,0)
#Change this line for a different model:
with slim.arg_scope(densenet.densenet_arg_scope(is_training=True)):
    #TODO: Check mobilenet_v1 module, var "excluding
    logits, _ = densenet.densenet121(image_bis, num_classes = len(labels_to_name))
variables = slim.get_variables_to_restore()
saver = tf.train.Saver(variables)
y = tf.nn.softmax(logits)
values, indices = tf.nn.top_k(y, 2)
table = tf.contrib.lookup.index_to_string_table_from_tensor(
          tf.constant([str(i) for i in range(2)]))
prediction_classes = table.lookup(tf.to_int64(indices))
with tf.Session() as sess:
  saver.restore(sess,  checkpoint_file)
  #TODO: "Noun of dataset"_"model"_"model version"
  export_path_base = "chestXray_densenet_121"
  export_path = os.path.join(
      tf.compat.as_bytes(export_path_base),
      tf.compat.as_bytes(str(FLAGS.model_version)))
  print('Exporting trained model to', export_path)
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)

  # Build the signature_def_map.
  classification_inputs = tf.saved_model.utils.build_tensor_info(
      serialized_tf_example)
  classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
      prediction_classes)
  classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

  classification_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={
              tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                  classification_inputs
          },
          outputs={
              tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                  classification_outputs_classes,
              tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                  classification_outputs_scores
          },
          method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

  tensor_info_x = tf.saved_model.utils.build_tensor_info(tf_example['x'])
  tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

  prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'images': tensor_info_x},
          outputs={'scores': tensor_info_y},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
          'predict_images':
              prediction_signature,
          tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              classification_signature,
      },
      legacy_init_op=legacy_init_op)

  builder.save()

  print('Done exporting!')