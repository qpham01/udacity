from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint  

import pickle

# Basic parameters
ksize = 3
kstride = 1
kpadding = "same"
kactivation = "relu"
psize = 2
dropout_prob = 0.4

def add_conv_pool_unit(model, filter_count, input_shape=None):
    if input_shape is not None:
        model.add(Conv2D(filters=filter_count, kernel_size=2, padding=kpadding, activation=kactivation, input_shape=input_shape))
    else:
        model.add(Conv2D(filters=filter_count, kernel_size=2, padding=kpadding, activation=kactivation))
    model.add(MaxPooling2D(pool_size=psize))
    
def load_data(file_name):
    with open(file_name, 'rb') as fin:
        tensors, targets = pickle.load(fin)
    return tensors, targets

def make_model():
    model = Sequential()

    ### TODO: Define your architecture.
    add_conv_pool_unit(model, 10, input_shape=(224, 224, 3))
    add_conv_pool_unit(model, 20)
    add_conv_pool_unit(model, 30)
    model.add(Flatten())
    model.add(Dense(5000, activation='relu'))
    model.add(Dropout(dropout_prob))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(133, activation='softmax'))
    model.summary()

    return model

### TODO: specify the number of epochs that you would like to use to train the model.
model = make_model()
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_tensors, train_targets = load_data('train.p')
valid_tensors, valid_targets = load_data('valid.p')
test_tensors, test_targets = load_data('test.p')
epochs = 20

### Do NOT modify the code below this line.

checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.from_scratch.hdf5', 
                               verbose=1, save_best_only=True)

model.fit(train_tensors, train_targets, 
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=40, callbacks=[checkpointer], verbose=1)
