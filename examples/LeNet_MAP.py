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
lenet       = models.LeNet(mnist_sample['x'])
set_determ  = lenet.set_deterministic()


#------------------------set up the loss for training--------------------------
loss_obj = losses.map_loss(lenet, tf.argmax(mnist_sample['y'], axis=-1))
m_loss   = loss_obj(range(len(lenet.sto_params))) #compute the loss based on all stochastic parameters
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100, 0.96, staircase=True)

# add all quantized parameters to the tf.summary
quant_vals = tf.get_collection('QUANT_VAL')
Quant_range_tags = ['Qunat_inputs','Quant_Weights_1','Quant_bias_1',
                    'Quant_Weights_2','Quant_bias_2',
                    'Quant_Weights_3','Quant_bias_3',
                    'Quant_Weights_4','Quant_bias_4',
                    'Quant_Weights_5','Quant_bias_5']
quant_range = tf.get_collection('QUANT_RANGE')
for idx,data in enumerate(quant_range):
    tf.summary.scalar(name=Quant_range_tags[idx], tensor=tf.reshape(data, []))

for qv in quant_vals:
    tf.summary.histogram(name=qv.name, values=qv)


#------------------------set up the optimizer---------------------------    
n_epochs    = 100
l_rate      = 1e-1
save_path   = "../results/exp_decay_lr"


#------------------------create the session and initialize--------------------------
sess = tf.Session(config=setup.config_tensorflow())
init = tf.global_variables_initializer()
sess.run(init)
sess.run(set_determ) #we need a deterministic mapping for MAP training

#create the tf.summary
accuracy        = None
loss_value      = None
learn_rate      = None
summary         = tf.summary.merge_all()
summary_writer  = tf.summary.FileWriter(save_path, sess.graph)
runtime_summary = tf.Summary()
runtime_summary.value.add(tag='accuracy', simple_value=accuracy)
runtime_summary.value.add(tag='loss', simple_value= loss_value)
runtime_summary.value.add(tag='learning_rate', simple_value= learn_rate)


#------------------------set up the optimizer (with gradient descend)--------------------------



# # Passing global_step to minimize() will increment it at each step.
# learning_step = (
# tf.train.GradientDescentOptimizer(learning_rate)
# .minimize(m_loss, global_step=global_step))
optimizer       = tf.train.GradientDescentOptimizer(learning_rate)
model_params    = tf.trainable_variables()
train           = optimizer.minimize(m_loss, var_list=model_params,global_step=global_step)

for ep in range(n_epochs):
    #re-initialize the iterator on the dataset
    sess.run(train_init)
    while True:
        try:
            #perfrom the update-step
            _, loss_value, learn_rate = sess.run((train, m_loss,learning_rate))
            
        except tf.errors.OutOfRangeError:
            break
        
    #evaluate the model on the training set
    sess.run(test_init)
    accuracy = evaluate_sample(lenet, mnist_sample, sess)
    print('Epoch %d, with loss %f, and accuracy %f' % (ep, loss_value, accuracy))

    runtime_summary.value[0].simple_value = accuracy
    runtime_summary.value[1].simple_value = loss_value
    runtime_summary.value[2].simple_value = learn_rate
    summary_writer.add_summary(runtime_summary, ep)

    #save the trained model parameters
    lenet.save_trainable(save_path+'_params.ckpt', sess)

