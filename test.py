import tensorflow as tf
import numpy as np
from alexnet import AlexNet
from dataspec import DataSpec
from scipy.misc import imread
from scipy.misc import imresize
import os.path as osp

import dataset


def display_results(image_paths, probs):
    '''Displays the classification results given the class probability for each image'''
    # Get a list of ImageNet class labels
    with open('imagenet-classes.txt', 'r') as infile:
        class_labels = list(map(str.strip, infile.readlines()))
    # Pick the class with the highest confidence for each image
    class_indices = np.argmax(probs, axis=1)
    # Display the results
    print('\n{:20} {:30} {}'.format('Image', 'Classified As', 'Confidence'))
    print('-' * 70)
    for img_idx, image_path in enumerate(image_paths):
        img_name = osp.basename(image_path)
        class_name = class_labels[class_indices[img_idx]]
        confidence = round(probs[img_idx, class_indices[img_idx]] * 100, 2)
        print('{:20} {:30} {} %'.format(img_name, class_name, confidence))

spec = DataSpec()
input_node = tf.placeholder(tf.float32, [None, 227, 227, 3])

net = AlexNet({"data": input_node})

# image = (imread(r"tank_1.jpg")[:, :, :3]).astype(np.float32)
# image -= np.mean(image)
# image = imresize(image, (227, 227, 3))
image_paths = [r"tank_1.jpg"]
image_producer = dataset.ImageProducer(image_paths=image_paths, data_spec=spec)

with tf.Session() as sess:
    # Start the image processing workers
    coordinator = tf.train.Coordinator()
    threads = image_producer.start(session=sess, coordinator=coordinator)

    # Load the converted parameters
    print('Loading the model')
    net.load(r"../AlexNet.npy", sess, encoding="latin1")

    # Load the input image
    print('Loading the images')
    indices, input_images = image_producer.get(sess)

    # Perform a forward pass through the network to get the class probabilities
    print('Classifying')
    print("prob:")
    probs = sess.run(net.get_output(), feed_dict={input_node: input_images})
    # probs是batch_size个预测结果
    top5 = probs[0].argsort()[::-1][:5]
    print(probs[0][top5])
    display_results([image_paths[i] for i in indices], probs)

    # Stop the worker threads
    coordinator.request_stop()
    coordinator.join(threads, stop_grace_period_secs=2)