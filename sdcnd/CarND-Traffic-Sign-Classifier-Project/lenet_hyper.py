"""
Contains hyper parameters
"""
import tensorflow as tf
from traffic_sign_data import COLOR_DEPTH

# Description of network
VERSION_DESCRIPTION = 'small network for traffic sign classification, l1 depth 25, l2 size 2500'

# Hyperparameters

# initial weight distribution
MU = 0
SIGMA = 0.1

# L2 regularization
L2_REG_STRENGTH = 1.0
BETA = 0.01

# traing hyper parameters
EPOCHS = [2]
BATCH_SIZES = [64]
LEARNING_RATE = 0.0001

# Place holders for features, labels, and keep probability
FEATURES = tf.placeholder(tf.float32, (None, 32, 32, COLOR_DEPTH))
LABELS = tf.placeholder(tf.int32, (None))
KEEP_PROB = tf.placeholder(tf.float32)
ONE_HOT_LABELS = tf.one_hot(LABELS, 43)
