"""
Implements a Bayesian LeNet with Gaussian parameters and trains it on the MNIST dataset, using MAP training.

by Lukas Mauch
"""

import tensorflow as tf
import bayesian_dnn.models as models
import bayesian_dnn.losses as losses
from bayesian_dnn.data import datasets
from bayesian_dnn.misc import setup
import numpy as np
from bayesian_dnn.evaluation import evaluate_sample

# ------------------------load the data and start a session---------------------------
batch_size = 1000
mnist_gen = datasets.Fashion_MNIST_Set()

# the training set:
mnist_train = mnist_gen(ds_format='cnn', key='train').shuffle(10000).batch(batch_size)
mnist_test = mnist_gen(ds_format='cnn', key='test').shuffle(10000).batch(batch_size)
mnist_iterator = tf.data.Iterator.from_structure(mnist_train.output_types, mnist_train.output_shapes)
train_init = mnist_iterator.make_initializer(mnist_train)
test_init = mnist_iterator.make_initializer(mnist_test)
mnist_sample = mnist_iterator.get_next()

# ------------------------create a model--------------------------
lenet = models.LeNet(mnist_sample['x'])
set_determ = lenet.set_deterministic()

# ------------------------set up the loss for training--------------------------
loss_obj = losses.map_loss(lenet, tf.argmax(mnist_sample['y'], axis=-1))
m_loss = loss_obj(range(len(lenet.sto_params)))  # compute the loss based on all stochastic parameters

# add all quantized parameters to the tf.summary

Quant_range_tags = ['Qunat_inputs', 'Quant_Weights_1', 'Quant_bias_1',
                    'Quant_Weights_2', 'Quant_bias_2',
                    'Quant_Weights_3', 'Quant_bias_3',
                    'Quant_Weights_4', 'Quant_bias_4',
                    'Quant_Weights_5', 'Quant_bias_5']
quant_range = tf.get_collection('QUANT_RANGE')
for idx, data in enumerate(quant_range):
    tf.summary.scalar(name=Quant_range_tags[idx], tensor=tf.reshape(data, []))
quant_vals = tf.get_collection('UnQuant_VAL')
print(quant_vals)
for idx,qv in enumerate(quant_vals):
    print(idx)
    tf.summary.histogram(name=qv.name, values=qv)

# ------------------------set up the optimizer---------------------------
n_epochs = 10
l_rate = 1e-1
save_path = "../results/Fashion_mnist/delete/"

# ------------------------create the session and initialize--------------------------
sess = tf.Session(config=setup.config_tensorflow())
init = tf.global_variables_initializer()
sess.run(init)
sess.run(set_determ)  # we need a deterministic mapping for MAP training

# create the tf.summary
accuracy = None
loss_value = None
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(save_path, sess.graph)
accuracy_summary = tf.Summary()
accuracy_summary.value.add(tag='accuracy', simple_value=accuracy)
accuracy_summary.value.add(tag='loss', simple_value=loss_value)

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
    accuracy = evaluate_sample(lenet, mnist_sample, sess)
    print('Epoch %d, with loss %f, and accuracy %f' % (ep, loss_value, accuracy))

    accuracy_summary.value[0].simple_value = accuracy
    accuracy_summary.value[1].simple_value = loss_value
    summary_writer.add_summary(accuracy_summary, ep)

    # save the trained model parameters
    lenet.save_trainable(save_path + '_params.ckpt', sess)

