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

# load and normalize mnist data
(mnist_train_x, mnist_train_y), (_, mnist_test_y) = tf.keras.datasets.mnist.load_data()
mnist_train_x = mnist_train_x / 255.

# load and normalize mnist-m data
with h5py.File(MNIST_M_PATH, "r") as mnist_m:
    mnist_m_train_x = mnist_m["train"]["X"][()] / 255. 
    mnist_m_test_x = mnist_m["test"]["X"][()] / 255. 
mnist_m_train_y = mnist_train_y
mnist_m_test_y = mnist_test_y

# prepare data
source = tf.data.Dataset.from_tensor_slices((mnist_train_x, mnist_train_y)).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE * 2)
target = tf.data.Dataset.from_tensor_slices((mnist_m_train_x, mnist_m_train_y)).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)
test = tf.data.Dataset.from_tensor_slices((mnist_m_test_x, mnist_m_test_y)).shuffle(SHUFFLE_SIZE).batch(BATCH_SIZE)

# cant iterate over a batch dataset so
# pass to np iterator and then form array

source_X = []
source_Y = []
target_X = []

iter = tf.data.Dataset.as_numpy_iterator(source)
for pair in iter:
    source_X.extend(pair[0][0])
    source_Y.extend(pair[1])

iter = tf.data.Dataset.as_numpy_iterator(target)
for pair in iter:
    target_X.extend(pair[0][0])

# since mnist-m data has RGB colour while mnist only
# has grayscale, dimensions of the two are mismatched.
# for training, duplicate grayscale values as though
# they were triples. then, flatten both the grayscale
# triples and RGB triples.

source_X = np.array(source_X)
source_X = np.repeat(source_X, 3, axis = 1)
source_Y = np.array(source_Y)
target_X = np.array(target_X)
target_X = np.reshape(target_X, (52500, 84))

dann = DANN()
dann.train(source_X, source_Y, target_X)
