"""
Implements a Bayesian LeNet with Gaussian parameters and trains it on the SVHN dataset, using MAP training.
"""

import tensorflow as tf
import bayesian_dnn.models as models
import bayesian_dnn.losses as losses
from bayesian_dnn.data import datasets
from bayesian_dnn.misc import setup
import numpy as np
from bayesian_dnn.evaluation import evaluate_sample

# ------------------------load the data and start a session---------------------------
batch_size = 512
svhn_gen = datasets.SVHN_set()

# the training set:
svhn_train = svhn_gen(key='train').shuffle(10000).batch(batch_size)
# the test set:
svhn_test = svhn_gen(key='test').shuffle(10000).batch(batch_size)
svhn_iterator = tf.data.Iterator.from_structure(svhn_train.output_types, svhn_train.output_shapes)
train_init = svhn_iterator.make_initializer(svhn_train)
test_init = svhn_iterator.make_initializer(svhn_test)
svhn_sample = svhn_iterator.get_next()

# ------------------------create a model--------------------------

svhn = models.SVHN(svhn_sample['x'])
set_determ = svhn.set_deterministic()

# ------------------------set up the loss for training--------------------------
loss_obj = losses.map_loss(svhn, tf.argmax(svhn_sample['y'], axis=-1))
m_loss = loss_obj(range(len(svhn.sto_params)))  # compute the loss based on all stochastic parameters

# ------------------------set up the optimizer---------------------------
n_epochs = 50
l_rate = 1e-1
save_path = "../results/svhn/map_training"

# ------------------------create the session and initialize--------------------------
sess = tf.Session(config=setup.config_tensorflow())
init = tf.global_variables_initializer()
sess.run(init)
sess.run(set_determ)  # we need a deterministic mapping for MAP training

# ------------------------set up the optimizer (with gradient descend)--------------------------
optimizer = tf.train.GradientDescentOptimizer(l_rate)
model_params = tf.trainable_variables()
train = optimizer.minimize(m_loss, var_list=model_params)

for ep in range(n_epochs):
    # re-initialize the iterator on the dataset
    sess.run(train_init)
    while True:
        try:
            # perfrom the update-step
            _, loss_value = sess.run((train, m_loss))

        except tf.errors.OutOfRangeError:
            break

    # evaluate the model on the training set
    sess.run(test_init)
    accuracy = evaluate_sample(svhn, svhn_sample, sess)
    print('Epoch %d, with loss %f, and accuracy %f' % (ep, loss_value, accuracy))

    # save the trained model parameters
    svhn.save_trainable(save_path + '_params.ckpt', sess)

