"""
Implements a Bayesian quantized QLeNet with Gaussian parameters and trains it on the MNIST dataset, using MAP training.

by Lukas Mauch
"""

import tensorflow as tf
import bayesian_dnn.models as models
import bayesian_dnn.losses as losses
from bayesian_dnn.data import datasets
from bayesian_dnn.misc import setup
import numpy as np
from bayesian_dnn.evaluation import evaluate_sample

#------------------------load the data and start a session---------------------------
batch_size      = 1000
mnist_gen       = datasets.MNIST_Set()

#the training set:
mnist_train     = mnist_gen(ds_format='cnn', key='train').shuffle(10000).batch(batch_size)
mnist_test      = mnist_gen(ds_format='cnn', key='test').shuffle(10000).batch(batch_size)
mnist_iterator  = tf.data.Iterator.from_structure(mnist_train.output_types, mnist_train.output_shapes) 
train_init      = mnist_iterator.make_initializer(mnist_train)
test_init       = mnist_iterator.make_initializer(mnist_test)
mnist_sample    = mnist_iterator.get_next()


#------------------------create a model--------------------------
lenet       = models.QLeNet(inp=mnist_sample['x'], m_init=tf.constant_initializer(0.01), k=2)
set_determ  = lenet.set_deterministic()


#------------------------set up the loss for training--------------------------
loss_obj = losses.map_loss(lenet, tf.argmax(mnist_sample['y'], axis=-1))
m_loss   = loss_obj(range(len(lenet.sto_params))) #compute the loss based on all stochastic parameters

#add all quantized parameters to the tf.summary
quant_vals = tf.get_collection('QUANT_VAL')
print(quant_vals)
# quant_range = tf.get_collection('QUANT_RANGE')
# print(quant_range)
# for qr in quant_range:
#     tf.summary.scalar(name='QUANT_RANGE',tensor=tf.convert_to_tensor(tf.reshape(qr,[])))
for qv in quant_vals:
    tf.summary.histogram(name=qv.name, values=qv)


#------------------------set up the optimizer---------------------------    
n_epochs    = 1
l_rate      = 1e-1
save_path   = "../results/qlenet/"


#------------------------create the session and initialize--------------------------
sess = tf.Session(config=setup.config_tensorflow())
init = tf.global_variables_initializer()
sess.run(init)
sess.run(set_determ) #we need a deterministic mapping for MAP training

#create the tf.summary
summary         = tf.summary.merge_all()
summary_writer  = tf.summary.FileWriter(save_path, sess.graph)

#------------------------set up the optimizer (with gradient descend)--------------------------
optimizer       = tf.train.GradientDescentOptimizer(l_rate)
model_params    = tf.trainable_variables()
train           = optimizer.minimize(m_loss, var_list=model_params)

for ep in range(n_epochs):
    #re-initialize the iterator on the dataset
    sess.run(train_init)
    while True:
        try:
            #perfrom the update-step
            _, loss_value, s = sess.run((train, m_loss, summary))

        except tf.errors.OutOfRangeError:
            break
    summary_writer.add_summary(s, ep)

    #evaluate the model on the training set
    sess.run(test_init)
    accuracy = evaluate_sample(lenet, mnist_sample, sess)
    print('Epoch %d, with loss %f, and accuracy %f' % (ep, loss_value, accuracy))

    #save the trained model parameters
    lenet.save_trainable(save_path+'_params.ckpt', sess)

