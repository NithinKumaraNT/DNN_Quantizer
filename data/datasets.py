import numpy as np
import tensorflow as tf
from keras.datasets import cifar10
import keras
import scipy as sp
import scipy.io as sio
from scipy.misc import *
from tensorflow.examples.tutorials.mnist import input_data

class MNIST_Set():
    def __call__(self, key='train',ds_format='dnn'):
        """
        loads the MNIST dataset and shuffle

        Returns:
            a tensorflow dataset
        """
        with tf.name_scope("MNIST_dataset"):
            mnist = tf.contrib.learn.datasets.load_dataset("mnist")
            if key=='train':
                if ds_format == 'dnn':
                    data    = mnist.train.images
                else:
                    data    = tf.reshape(mnist.train.images,[-1,28,28,1])
                labels  = np.asarray(mnist.train.labels)
            else:
                if ds_format == 'dnn':
                    data    = mnist.test.images
                else:
                    data    = tf.reshape(mnist.test.images,[-1,28,28,1])
                labels = np.asarray(mnist.test.labels)

            return tf.data.Dataset.from_tensor_slices({"x": data, "y": tf.one_hot(labels, depth=10)})

class CIFAR10_Set():
    def __call__(self, key='train'):
        """
        loads the CIFAR10 dataset and shuffle

        Returns:
            a tensorflow dataset
        """
        with tf.name_scope("CIFAR10_dataset"):
            (x_train, y_train), (x_test, y_test) = cifar10.load_data()
            x_train, x_test = self.normalize(x_train, x_test)
            if key == 'train':
                data = x_train.astype('float32')
                labels = keras.utils.to_categorical(y_train, 10)
            else:
                data = x_test.astype('float32')
                labels = keras.utils.to_categorical(y_test, 10)
            return tf.data.Dataset.from_tensor_slices({"x": data, "y": labels})

    def normalize(self, X_train, X_test):
        # this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train, axis=(0, 1, 2, 3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train - mean) / (std + 1e-7)
        X_test = (X_test - mean) / (std + 1e-7)

        return X_train, X_test



class Fashion_MNIST_Set():
    def __call__(self, key='train', ds_format='dnn'):
        """
        loads the Fashion MNIST dataset and shuffle

        Returns:
            a tensorflow dataset
        """
        mnist = input_data.read_data_sets('data/fashion')
        if key == 'train':
            if ds_format == 'dnn':
                data = mnist.train.images
            else:
                data = tf.reshape(mnist.train.images, [-1, 28, 28, 1])
            labels = np.asarray(mnist.train.labels)
        else:
            if ds_format == 'dnn':
                data = mnist.test.images
            else:
                data = tf.reshape(mnist.test.images, [-1, 28, 28, 1])
            labels = np.asarray(mnist.test.labels)

        return tf.data.Dataset.from_tensor_slices({"x": data, "y": tf.one_hot(labels, depth=10)})

class SVHN_set():
    def __call__(self, key='train'):
        """
        loads the SVHN dataset and shuffle

        Returns:
            a tensorflow dataset
        """

        # load data
        data_train_dict = sio.loadmat(r'C:\Users\NNR5KOR\Thesis\ParserProject\bayesian_dnn\data\svhn\train_32x32.mat')
        x_train = data_train_dict["X"]
        y_train = data_train_dict["y"] - 1
        data_test_dict = sio.loadmat(r'C:\Users\NNR5KOR\Thesis\ParserProject\bayesian_dnn\data\svhn\test_32x32.mat')
        x_test = data_test_dict["X"]
        y_test = data_test_dict["y"] - 1
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train = np.reshape(x_train.T, (-1, 3 * 32 * 32))
        x_test = np.reshape(x_test.T, (-1, 3 * 32 * 32))

        # normalize data
        m_train = np.mean(x_train, axis=0)
        std_train = np.std(x_train, axis=0)
        data_train = (x_train - m_train) / std_train
        m_test = np.mean(x_test, axis=0)
        std_test = np.std(x_test, axis=0)
        data_test = (x_test - m_test) / std_test

        # convert class vectors to binary class matrices
        label_train = list()
        for val in y_train:
            tmp = np.zeros(10)
            tmp[val] = 1
            label_train.append(tmp)
        label_test = list()
        for val in y_test:
            tmp = np.zeros(10)
            tmp[val] = 1
            label_test.append(tmp)

        if key=='train':
            # shape back to rgb image if vectorized==False
            data = tf.reshape(data_train, [-1, 32, 32, 3])
            labels = np.asarray(label_train)

        else:
            data = tf.reshape(data_test, [-1, 32, 32, 3])
            labels = np.asarray(label_test)

        return tf.data.Dataset.from_tensor_slices({"x": data, "y": labels})


#----------------------------------TEST----------------------------------
if __name__ == "__main__":
    dataset = CIFAR10_Set()
    dataset()
