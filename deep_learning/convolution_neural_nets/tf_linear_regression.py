import tensorflow as tf
import numpy as np


X_train = np.asarray([1,2.2,3.3,4.1,5.2])
Y_train =  np.asarray([2,3,3.3,4.1,3.9,1.6])

def model(X, w):
    return tf.mul(X, w)


X = tf.placeholder("float") # create symbolic variables
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weights")
y_model = model(X, w) # our predicted values

cost = tf.pow(Y-y_model, 2) # squared error cost

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost) #sgd optimization
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

print('w before: ', sess.run(w))
for trials in range(20):  #
    for (x, y) in zip(X_train, Y_train):
        t_op, c, iw = sess.run([train_op, cost, w], feed_dict={X: x, Y: y})
    #print('w ', iw, 'at step ', trials)
    print('sesw ', sess.run(w), 'at step ', trials)
    print('cost ', c, 'at step ', trials)

print('w after: ', sess.run(w))