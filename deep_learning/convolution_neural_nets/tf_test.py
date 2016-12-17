import tensorflow as tf

x = tf.constant([22,21,32], name='x')
d=tf.constant([12,23,43],name='d')
y = tf.Variable(x * d, name='y')
model = tf.initialize_all_variables()
"""
with tf.Session() as session:    
    session.run(model)
    r = session.run(y)    
    print(r)
"""
sess = tf.Session()
sess.run(model)
r = sess.run(y)
print(r)
   
a = tf.placeholder(tf.int32)
b = tf.placeholder(tf.int32)
sess = tf.Session()
r = sess.run(a+b, feed_dict={a: 111, b: 222})
print(r)

matrix1 = tf.constant([[1, 2,32], [3, 4,2],[3,2,11]])
matrix2 = tf.constant([[21,3,12], [3, 56,2],[35,21,61]])
product = tf.matmul(matrix1, matrix2)
with tf.Session() as sess:
    result = sess.run(product)
    print(result)