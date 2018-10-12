import tensorflow as tf
slim = tf.contrib.slim
from utils.utils_csv import _get_infos
import research.slim.nets.mobilenet.mobilenet_v2 as mobilenet_v2
import DenseNet.preprocessing.densenet_pre as dp
import cv2
import os
import numpy as np
dataset_dir= "D:/MURA-v1.1"
checkpoint_dir = "D:/ckpt-mura.05.08.2018"
checkpoint_file = os.path.join(checkpoint_dir,"model-115001")

image_size = 224
#Labels 
labels_to_name = {0:'negative', 
                1:'positive'
                }

tf.logging.set_verbosity(tf.logging.INFO)
path_img = ""
file_input = tf.placeholder(tf.string, ())
image = tf.image.decode_image(tf.read_file(file_input), channels=3)
copy = tf.identity(image)
image = tf.image.convert_image_dtype(image, tf.float32)
image.set_shape([None,None,3])
image_a = dp.preprocess_image(image, 224,224, is_training=False)
images_bis = tf.expand_dims(image_a,0)
with slim.arg_scope(mobilenet_v2.training_scope(is_training=False)):
    #TODO: Check mobilenet_v1 module, var "excluding
    logits, end_points = mobilenet_v2.mobilenet(images_bis,depth_multiplier=1.4, num_classes = len(labels_to_name))
variables = slim.get_variables_to_restore()
init_fn=slim.assign_from_checkpoint_fn(checkpoint_file, variables)
embedding = end_points["layer_18/output"][0]
weights = tf.reduce_mean(tf.get_default_graph().get_tensor_by_name("MobilenetV2/expanded_conv_16/project/weights:0"), axis=[0,1,2])

with tf.Session() as sess:
    init_fn(sess)
    emb, ng, raw_img = sess.run([embedding, weights,image], feed_dict={file_input:path_img})
    cam = np.zeros(emb.shape[0 : 2], dtype = np.float32)
    for j, w in enumerate(ng):
        cam +=  emb[:, :, j]*w
    cam /= np.max(cam)
    cam = cv2.resize(cam, (raw_img.shape[1],raw_img.shape[0]))
    cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    alpha = 0.0015
    new_img = raw_img+alpha*cam
    cv2.imwrite("cam.png",np.uint8(255*new_img))
