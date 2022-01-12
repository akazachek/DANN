from DANN import DANN
import h5py
import tensorflow as tf
import numpy as np

"""
    MNIST-M dataset prepared by:
    https://github.com/sghoshjr/Domain-Adversarial-Neural-Network/blob/master/DANN.py
"""  

MNIST_M_PATH = "./../Datasets/MNIST-M/mnistm.h5"
SHUFFLE_SIZE = 1000
BATCH_SIZE = 32
CHANNELS = 3

### prepare data

# load and normalize mnist data
(mnist_train_x, mnist_train_y), (_, mnist_test_y) = tf.keras.datasets.mnist.load_data()
mnist_train_x = mnist_train_x / 255.

# load and normalize mnist-m data
with h5py.File(MNIST_M_PATH, "r") as mnist_m:
    mnist_m_train_x = mnist_m["train"]["X"][()] / 255. 
    mnist_m_test_x = mnist_m["test"]["X"][()] / 255. 
mnist_m_train_y = mnist_train_y
mnist_m_test_y = mnist_test_y

# honestly batching might be unnecessary here.
source = tf.data.Dataset.from_tensor_slices((mnist_train_x, mnist_train_y)).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE * 2)
target = tf.data.Dataset.from_tensor_slices((mnist_m_train_x, mnist_m_train_y)).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
test = tf.data.Dataset.from_tensor_slices((mnist_m_test_x, mnist_m_test_y)).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)

### convert data to NP arrays

source_X = []
source_Y = []
target_X = []

for batch in source:
    source_X.extend(batch[0])
    source_Y.extend(batch[1])

for batch in target:
    target_X.extend(batch[0])

# since mnist-m data has RGB colour while mnist only
# has grayscale, dimensions of the two are mismatched.
# for training, duplicate grayscale values as though
# they were triples. then, flatten both the grayscale
# triples and RGB triples.

source_X = np.array(source_X)
source_X = np.reshape(source_X, (60000, 28*28))
source_X = np.repeat(source_X, CHANNELS, axis = 1)
source_Y = np.array(source_Y)

target_X = np.array(target_X)
target_X = np.reshape(target_X, (60000, 28*28*CHANNELS))

dann = DANN(max_iter = 10)
dann.train(source_X, source_Y, target_X)

### test accuracy

test_X = []
test_Y = []

for batch in test:
    test_X.extend(batch[0])
    test_Y.extend(batch[1])

test_X = np.array(test_X)
test_X = np.reshape(test_X, (10000, 28*28*CHANNELS))
test_Y = np.array(test_Y)

# note that successful predictions will have a
# difference of zero
predictions = dann.predict_labels(test_X)
diffs = predictions - test_Y

num_input = len(test_Y)
num_success = np.count_nonzero(diffs == 0)
print("Number of successful predictions: {}".format(num_success))
print("Number of unsuccessful predictions: {}".format(num_input - num_success))
print("Overall accuracy: {}%".format(num_success / num_input * 100))

