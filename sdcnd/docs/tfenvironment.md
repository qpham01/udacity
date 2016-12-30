# Creating the TensorFlow Environment with GPU Support

First install CUDA Toolkit version 8.0 and CuDNN v5 from NVidia for 64-bit Ubuntu.  See elsewhere for how to do this.

## OS X and Linux

Install Anaconda

This lab requires Anaconda and Python 3.4 or higher. If you don't meet all of these requirements, install the appropriate package(s).

### Run the Anaconda Environment

Follow these steps to setup the Udacity TensorFlow environment:

1. Download the zip file of the repository https://github.com/udacity/CarND-TensorFlow-Lab.git.
2. Unzip it in a good place in your own source repository.
3. Change the environment name in the first line of environment.yml to something short, like utf.
4. Run the following commands to create the environment, activate the environment, and install TensorFlow with GPU support in that new environment.
```shell
$ cd [path]/CarND-TensorFlow-Lab-master
$ conda env create -f environment.yml
$ source activate utf
(utf)$ wget https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
(utf)$ pip3 install --upgrade tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
```

Run the following python script to verify that TensorFlow with GPU support is active in your environment.
```py3
import tensorflow as tf

HELLO_CONSTANT = tf.constant('Hello World!')

with tf.Session() as sess:
    # Run the tf.constant operation in the session
    OUTPUT = sess.run(HELLO_CONSTANT)
    print(OUTPUT)
```