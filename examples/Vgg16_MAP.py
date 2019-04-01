"""
Implements a CIFAR_10 with VGG16 architecture and trains it on the CIFAR_10 dataset, using MAP training.

by Nithin Kumara N T
"""

import tensorflow as tf
import bayesian_dnn.models as models
import bayesian_dnn.losses as losses
from bayesian_dnn.data import datasets
from bayesian_dnn.misc import setup
import numpy as np
from bayesian_dnn.evaluation import evaluate_sample

# ------------------------load the data and start a session---------------------------
batch_size = 128
cifar10_gen = datasets.CIFAR10_Set()

# the training set:
cifar10_train = cifar10_gen(key='train').shuffle(10000).batch(batch_size)
cifar10_test = cifar10_gen(key='test').shuffle(10000).batch(batch_size)
cifar10_iterator = tf.data.Iterator.from_structure(cifar10_train.output_types, cifar10_train.output_shapes)
train_init = cifar10_iterator.make_initializer(cifar10_train)
test_init = cifar10_iterator.make_initializer(cifar10_test)
cifar10_sample = cifar10_iterator.get_next()

# ------------------------create a model--------------------------
cifar10_net = models.Vgg16_Cifar10(cifar10_sample['x'])
set_determ = cifar10_net.set_deterministic()

# ------------------------set up the loss for training--------------------------
loss_obj = losses.map_loss(cifar10_net, tf.argmax(cifar10_sample['y'], axis=-1))
m_loss = loss_obj(range(len(cifar10_net.sto_params)))  # compute the loss based on all stochastic parameters
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100, 0.96, staircase=True)


# ------------------------set up the optimizer---------------------------
n_epochs = 100
l_rate = 1e-1
save_path = "../results/delete/Test_vgg16/VGG16_MAP"

# ------------------------create the session and initialize--------------------------
sess = tf.Session(config=setup.config_tensorflow())
init = tf.global_variables_initializer()
sess.run(init)
sess.run(set_determ)  # we need a deterministic mapping for MAP training

# print(np.sqrt(sess.run(tf.nn.moments(tf.reshape(cifar10_net.W1.var_dist.loc, [-1 ,1]), axes=[0]))[-1]))
# print(np.sqrt(sess.run(tf.nn.moments(tf.reshape(cifar10_net.W2.var_dist.loc, [-1 ,1]), axes=[0]))[-1]))
# print(np.sqrt(sess.run(tf.nn.moments(tf.reshape(cifar10_net.W3.var_dist.loc, [-1 ,1]), axes=[0]))[-1]))
# print(np.sqrt(sess.run(tf.nn.moments(tf.reshape(cifar10_net.W4.var_dist.loc, [-1 ,1]), axes=[0]))[-1]))
# print(np.sqrt(sess.run(tf.nn.moments(tf.reshape(cifar10_net.W5.var_dist.loc, [-1 ,1]), axes=[0]))[-1]))

# create the tf.summary
accuracy = None
loss_value = None
learn_rate = None
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(save_path, sess.graph)
runtime_summary = tf.Summary()
runtime_summary.value.add(tag='accuracy', simple_value=accuracy)
runtime_summary.value.add(tag='loss', simple_value=loss_value)
runtime_summary.value.add(tag='learning_rate', simple_value=learn_rate)


# ------------------------set up the optimizer (with gradient descend)--------------------------


# # Passing global_step to minimize() will increment it at each step.
# learning_step = (
# tf.train.GradientDescentOptimizer(learning_rate)
# .minimize(m_loss, global_step=global_step))
optimizer = tf.train.GradientDescentOptimizer(l_rate)
model_params = tf.trainable_variables()
train = optimizer.minimize(m_loss, var_list=model_params, global_step=global_step)

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
    accuracy = evaluate_sample(cifar10_net, cifar10_sample, sess)
    print('Epoch %d, with loss %f, and accuracy %f' % (ep, loss_value, accuracy))

    runtime_summary.value[0].simple_value = accuracy
    runtime_summary.value[1].simple_value = loss_value
    # runtime_summary.value[2].simple_value = learn_rate
    summary_writer.add_summary(runtime_summary, ep)

    # save the trained model parameters
    cifar10_net.save_trainable(save_path + '_params.ckpt', sess)

