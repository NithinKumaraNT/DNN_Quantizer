"""
Implements a Bayesian LeNet with Gaussian parameters and trains it on the MNIST dataset, using SGVB training.

by Lukas Mauch
"""

import tensorflow as tf
import models
import losses
from data import datasets
from misc import setup
import numpy as np
from evaluation import evaluate_sample

#------------------------load the data and start a session---------------------------
batch_size      = 5000
mnist_gen       = datasets.MNIST_Set()

#the training set:
mnist_train     = mnist_gen(ds_format='cnn', key='train').shuffle(10000).batch(batch_size)
mnist_test      = mnist_gen(ds_format='cnn', key='test').shuffle(10000).batch(batch_size)
mnist_iterator  = tf.data.Iterator.from_structure(mnist_train.output_types, mnist_train.output_shapes) 
train_init      = mnist_iterator.make_initializer(mnist_train)
test_init       = mnist_iterator.make_initializer(mnist_test)
mnist_sample    = mnist_iterator.get_next()


#------------------------create a model--------------------------
lenet = models.LeNet(mnist_sample['x'])


#------------------------set up the loss for training--------------------------
loss_obj = losses.variational_loss(lenet, tf.argmax(mnist_sample['y'], axis=-1))
v_loss   = loss_obj(range(len(lenet.sto_params))) #compute the loss based on all stochastic parameters


#------------------------set up the optimizer---------------------------    
n_epochs    = 100
l_rate      = 1e-4
save_path   = "../results/lenet/sgvb_training"


#------------------------create the session and initialize--------------------------
sess = tf.Session(config=setup.config_tensorflow())
init = tf.global_variables_initializer()
sess.run(init)


#------------------------set up the optimizer (with gradient descend)--------------------------
optimizer       = tf.train.GradientDescentOptimizer(l_rate)
model_params    = tf.trainable_variables()
train           = optimizer.minimize(v_loss, var_list=model_params)

for ep in range(n_epochs):
    #re-initialize the iterator on the dataset
    sess.run(train_init)
    while True:
        try:
            #perfrom the update-step
            _, loss_value = sess.run((train, v_loss))
            
        except tf.errors.OutOfRangeError:
            break
        
    #evaluate the model on the training set
    sess.run(test_init)
    accuracy = evaluate_sample(lenet, mnist_sample, sess)
    print('Epoch %d, with loss %f, and accuracy %f' % (ep, loss_value, accuracy))

    #save the trained model parameters
    lenet.save_trainable(save_path+'_params.ckpt', sess)

