"""
from keras.models import load_model
from keras.utils.visualize_util import plot

model = load_model('model.h5')
plot(model, to_file='model.png')
"""
from model import load_data

images, labels = load_data('train.p')

print(images.shape)