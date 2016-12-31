"""
Demonstrating the effects of TensorFlow session runs
"""
import tensorflow as tf

output = None

# Input
features = tf.Variable([1.0, -3.0])

# Create Model
hidden_layer1 = tf.add(features, [2.0, 2.0])
hidden_layer2 = tf.nn.relu(hidden_layer1)
output = tf.add(hidden_layer2, [3.0, 3.0])

# Create an operation that initializes all variables
init = tf.initialize_all_variables()

with tf.Session() as session:
    out = session.run(init)
    print("-----------")
    print("Running a session to initialize variable has no output")
    print(out)
    hl1, hl2 = session.run((hidden_layer1, hidden_layer2))
    print("-----------")
    print("Running a session after initialization, asking for hidden layers")
    print(hl1)
    print("Note that the ReLU applied to produce hidden_layer2 changed the -1 to 0")
    print(hl2)
    out = session.run(output)
    print("-----------")
    print("Running a session after initialization, asking for final output")
    print("TF will compute all variables needed to compute the requested output.")
    print("The hidden_layer will be computed before computing output because the")
    print("latter depends on the former in the tensor graph, just like in miniflow")
    print(out)
    print("-----------")
    out = session.run(output)
    print("And again after initialization, asking for same, will output the same value")
    print(out)
    print("That is, after initialization, each session will compute the requested output,")
    print("based on the operations that has been set up using the tf.methods() using the")
    print("originally initialized variable values")
    print("-----------")
