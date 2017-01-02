"""
Sandbox for running script snippets as experiments
"""
import tensorflow as tf

n_features = 5
n_labels = 4
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))

with tf.Session() as session:
    init = tf.initialize_all_variables()
    session.run(init)
    output = session.run(weights)

print(output)