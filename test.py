import tensorflow as tf

x = tf.placeholder(tf.float32, name="input")
# y =a*x + b 
a = tf.constant(2., dtype=tf.float32)
b = tf.constant(0.5,dtype=tf.float32 )
y = a*x+b

with tf.Session() as sess:
    c = sess.run([y], feed_dict={'input:0' : 0.2})
    print(c)