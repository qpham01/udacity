def parallel_overlap_count(data1, data2, threshold = 0):
      n_data1 = len(data1)
  n_data2 = len(data2)
  n_total = n_data1 * n_data2
  n_overlap = 0
  n_compare = 0
  n_process = 0
  max_threads = 14
  i_start = 0  
  while (n_process < n_data1):
    i_end = i_start + max_threads - 1        
    # ex_data1 = pymp.shared.array(data1[i_start:i_end,:,:]) 
    ex_counters = pymp.shared.array((2,), dtype='uint32') 
    with pymp.Parallel(max_threads) as p:
      for i in p.range(i_start, max_threads):
        for j in xrange(n_data2):
          if (is_similar(data1[i], data2[j])):
            #with p.lock:
            n_overlap += 1
              #print ("found ", ex_counters[0], " overlaps")
          with p.lock:
            n_compare += 1
          #if (ex_counters[1] % 1000 == 0):
          #    print("Made ", ex_counters[1], " processes out of ", n_data1)
    i_start = i_end + 1
  return n_overlap


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)

y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))