### Installing Keras with TensorFlow and GPU support

These instructions are for the Anaconda3 environment.

First make sure tensorflow with gpu is installed:

    pip install -I tensorflow-gpu

Then install Keras

    pip install -I keras

Make sure that your .keras/keras.json file looks like this:
```
{
    "floatx": "float32",
    "image_dim_ordering": "tf",
    "epsilon": 1e-07,
    "backend": "tensorflow"
}
```

Run the following:

    $ python

The following should be printed out:

    Python 3.5.2 |Anaconda custom (64-bit)| (default, Jul  2 2016, 17:53:06) 
    [GCC 4.4.7 20120313 (Red Hat 4.4.7-1)] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>> 

Type in:
    
    >>> import keras

The following should be printed out:

    Using TensorFlow backend.
    I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcublas.so locally
    I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcudnn.so locally
    I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcufft.so locally
    I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcuda.so.1 locally
    I tensorflow/stream_executor/dso_loader.cc:128] successfully opened CUDA library libcurand.so locally

