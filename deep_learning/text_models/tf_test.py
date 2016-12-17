import tensorflow as tf

num_nodes = 4

graph = tf.Graph()
with graph.as_default():
  ix = tf.Variable(tf.truncated_normal([5, 5], -0.1, 0.1))
  im = tf.Variable(tf.truncated_normal([num_nodes, num_nodes], -0.1, 0.1))
  ib = tf.Variable(tf.zeros([1, num_nodes]))

# Truncated normal
with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print (session.run(ix))
  print (session.run(im))
  print (session.run(ib))   