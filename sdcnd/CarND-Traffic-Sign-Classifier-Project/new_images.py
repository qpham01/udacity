"""
Read and show new images
"""
from matplotlib import image as mpimg
import numpy as np
import tensorflow as tf
from traffic_sign_model import evaluate

i = 1
fname = 'sign{:02d}.png'.format(i)
images = mpimg.imread(fname)
images = np.expand_dims(images, axis=0)
print("Image shape", images.shape)
for i in range(2,7):
    fname = 'sign{:02d}.png'.format(i)
    img = mpimg.imread(fname)
    img = np.expand_dims(img, axis=0)
    print("image %s shape" % (i), img.shape)
    images = np.append(images, img, axis=0)

#printing out some stats and plotting
print('The extra data shape is:', images.shape)

y_extra = [14,28,13,27,17,26]

saver = tf.train.Saver()

save_file = 'traffic_signs.ckpt'

# Evaluate the model with test data
with tf.Session() as sess:
    saver.restore(sess, save_file)

    test_accuracy = evaluate(images, y_extra, 6)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
