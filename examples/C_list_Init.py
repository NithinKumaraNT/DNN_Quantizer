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
from bayesian_dnn.evaluation import evaluate_sample_xchange
from bayesian_dnn.quantization import ClippedUniformQuantizer as ClippedUniformQuantizer
from bayesian_dnn.quantization import ClippedApproxUniformQuantizer as ClippedApproxUniformQuantizer
tf.reset_default_graph()


#------------------------load the data and start a session---------------------------
batch_size      = 1000
mnist_gen       = datasets.MNIST_Set()

#the training set:
with tf.name_scope("iterators"):
    mnist_train     = mnist_gen(ds_format='cnn', key='train').shuffle(10000).batch(batch_size)
    mnist_test      = mnist_gen(ds_format='cnn', key='test').shuffle(10000).batch(batch_size)
    mnist_iterator  = tf.data.Iterator.from_structure(mnist_train.output_types, mnist_train.output_shapes)
    train_init      = mnist_iterator.make_initializer(mnist_train)
    test_init       = mnist_iterator.make_initializer(mnist_test)
    mnist_sample    = mnist_iterator.get_next()


#------------------------create a model--------------------------
#--Approx quantizer model---

range_init_list = [tf.constant_initializer(0.91), #quant_input
tf.constant_initializer(0.2), #quant_weights_1
tf.constant_initializer(1.93), #quant_activation_1
tf.constant_initializer(0.1), #quant_weights_2
tf.constant_initializer(1.76), #quant_activation_2
tf.constant_initializer(0.1), #quant_weights_3
tf.constant_initializer(1.93), #quant_activation_3
tf.constant_initializer(0.2), #quant_weights_4
tf.constant_initializer(3.42), #quant_activation_4
tf.constant_initializer(0.25)  #quant_weights_5
]

lenet       =models.QC_approx_Uni_LeNet(inp=mnist_sample['x'],c_init_list= range_init_list,k=2,n_steps= 24) # c_init is intialized internally

set_determ  = lenet.set_deterministic()
with tf.name_scope("Decay_learn_rate"):
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.1
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                               100, 0.96, staircase=True)


#------------------------set up the loss for training--------------------------
loss_obj = losses.map_loss(lenet, tf.argmax(mnist_sample['y'], axis=-1))
m_loss   = loss_obj(range(len(lenet.sto_params))) #compute the loss based on all stochastic parameters

# with tf.variable_scope("test_model"):
#     test_lenet =main_model.QHardLeNet(inp=mnist_sample['x'], c_init=tf.constant_initializer(0.1))


# add all quantized parameters to the tf.summary
quant_vals_Uniform = tf.get_collection('QUANT_VAL_UNIFORM')
quant_vals_Approx = tf.get_collection('QUANT_VAL_APPROX')
quant_range = tf.get_collection('QUANT_RANGE')
for data in quant_range:
    tf.summary.scalar(name=data.name, tensor=tf.reshape(data, []))

for qv in quant_vals_Uniform:
    tf.summary.histogram(name=qv.name, values=qv)
for qv in quant_vals_Approx:
    tf.summary.histogram(name=qv.name, values=qv)
#


#------------------------set up the optimizer---------------------------
n_epochs    = 20
l_rate      = 1e-1
save_path   = "../results/exchange/steps_20"


#------------------------create the session and initialize--------------------------
sess = tf.Session(config=setup.config_tensorflow())
init = tf.global_variables_initializer()
sess.run(init)
sess.run(set_determ) #we need a deterministic mapping for MAP training

# print(np.sqrt(sess.run(tf.nn.moments(tf.reshape(lenet.W1.var_dist.loc, [-1 ,1]), axes=[0]))[-1]))
# print(np.sqrt(sess.run(tf.nn.moments(tf.reshape(lenet.W2.var_dist.loc, [-1 ,1]), axes=[0]))[-1]))
# print(np.sqrt(sess.run(tf.nn.moments(tf.reshape(lenet.W3.var_dist.loc, [-1 ,1]), axes=[0]))[-1]))
# print(np.sqrt(sess.run(tf.nn.moments(tf.reshape(lenet.W4.var_dist.loc, [-1 ,1]), axes=[0]))[-1]))
# print(np.sqrt(sess.run(tf.nn.moments(tf.reshape(lenet.W5.var_dist.loc, [-1 ,1]), axes=[0]))[-1]))



#create the tf.summary
accuracy        = None
loss_value      = None
learn_rate      = None
accuracy_uni    = None
summary         = tf.summary.merge_all()
summary_writer  = tf.summary.FileWriter(save_path, sess.graph)
runtime_summary = tf.Summary()
runtime_summary.value.add(tag='accuracy', simple_value=accuracy)
runtime_summary.value.add(tag='loss', simple_value= loss_value)
runtime_summary.value.add(tag='learning_rate', simple_value= learn_rate)
runtime_summary.value.add(tag='accuracy_uni', simple_value= accuracy_uni)

#------------------------set up the optimizer (with gradient descend)--------------------------
optimizer       = tf.train.GradientDescentOptimizer(l_rate)
model_params    = tf.trainable_variables()
train           = optimizer.minimize(m_loss, var_list=model_params,global_step=global_step)

for ep in range(n_epochs):
    #re-initialize the iterator on the dataset
    sess.run(train_init)
    while True:
        try:
            #perfrom the update-step
            _, loss_value, s, learn_rate= sess.run((train, m_loss, summary,learning_rate))
        except tf.errors.OutOfRangeError:
            break



    #evaluate the model on the training set
    sess.run(test_init)
    # accuracy = evaluate_sample(lenet, mnist_sample, sess)
    accuracy,accuracy_uni = evaluate_sample_xchange(lenet, mnist_sample, sess)
    print('Epoch %d, with loss: %f, approx_quant_accuracy: %f and uniform_quant_accuracy: %f' % (ep, loss_value, accuracy,accuracy_uni))
    # print('Epoch %d, with loss: %f, accuracy: %f ' % (ep, loss_value, accuracy))

    runtime_summary.value[0].simple_value = accuracy
    runtime_summary.value[1].simple_value = loss_value
    runtime_summary.value[2].simple_value = learn_rate
    runtime_summary.value[3].simple_value = accuracy_uni
    summary_writer.add_summary(runtime_summary, ep)

    # Write to summary writer
    summary_writer.add_summary(s, ep)
    #save the trained model parameters

    lenet.save_trainable(save_path+'\map_training_params.ckpt', sess)

    # tf.reset_default_graph()


