""" Tests of tensorflow ops """
import tensorflow as tf

batch_size = 2
seq_length = 3
'''
with tf.Graph().as_default():
    new_slice = tf.Variable(tf.int32, [batch_size, batch_size])
    value = tf.placeholder(tf.int32)
    target_data = tf.placeholder(tf.int32, [batch_size, seq_length])
    new_slice.assign(target_data[0:batch_size, 0:batch_size])
    new_slice[0, 0].assign(value)

    test_target_data = [[10, 20, 30], [40, 18, 23]]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(new_slice, {target_data: test_target_data, value: 3})
        print(output)
'''
go_id = 358

with tf.Graph().as_default():
    target_data = tf.placeholder(tf.int32, [batch_size, seq_length])
    first_column = tf.fill([batch_size, 1], go_id)
    no_last_word = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    with_first_column = tf.concat([first_column, no_last_word], 1)

    test_target_data = [[10, 20, 30], [40, 18, 23]]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output = sess.run(target_data, {target_data: test_target_data})
        print(output)
        output = sess.run(first_column, {target_data: test_target_data})
        print(output)
        output = sess.run(no_last_word, {target_data: test_target_data})
        print(output)
        #output = sess.run(with_first_column, {target_data: test_target_data})
        #print(output)        
