import pickle
import numpy as np

def load_data(data_file):
    data = pickle.load(open(data_file, "rb"))
    images = data["images"]
    labels = data["labels"]

    return images, labels

images04, labels04 = load_data('train.p.04')
images05, labels05 = load_data('train.p.05')
images06, labels06 = load_data('train.p.06')
full_length = images04.shape[0] + images05.shape[0] + images06.shape[0]

big = np.zeros((full_length, 160, 320, 1))

print(big.shape)



def add_array(big, small, offset = 0):
    for i in range(small.shape[0]):
        big[i + offset] = small[i]

n = 0
add_array(big, images04, n)
labels = labels04
n += images04.shape[0]
del images04
del labels04

add_array(big, images05, n)
labels.extend(labels05)
n += images05.shape[0]
del images05
del labels05

add_array(big, images06, n)
labels.extend(labels06)
del images06
del labels06

data_pickle = {}
data_pickle["images"] = big
data_pickle["labels"] = labels
pickle.dump(data_pickle, open('train.p', "wb"))