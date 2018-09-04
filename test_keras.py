import tensorflow as tf
import h5py
model = tf.keras.applications.densenet.DenseNet121(classes=14, include_top=True, weights=None)
model.load_weights("C:/Users/User12/Documents/chest_hd5/chest_weights.h5")
saver = tf.train.Saver()
sess = tf.keras.backend.get_session()
save_path = saver.save(sess, "./chest_densenet.ckpt")