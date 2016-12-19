import tensorflow as tf
import numpy as np
from alexnet import AlexNet
from dataspec import DataSpec
from scipy.misc import imread
from scipy.misc import imresize

spec = DataSpec()
input_node = tf.placeholder(tf.float32, [None, 227, 227, 3])

net = AlexNet({"data": input_node})

image = (imread(r"tank_1.jpg")[:, :, :3]).astype(np.float32)
image -= np.mean(image)
image = imresize(image, (227, 227, 3))

with tf.Session() as sess:
    net.load("AlexNet.npy", sess)
    print("prob:")
    probs = sess.run(net.get_output(), feed_dict={input_node: [image, image]})