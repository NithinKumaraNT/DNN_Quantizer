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
#--Approx quantizer model---

range_init_list = [tf.constant_initializer(2.1), #quant_input
tf.constant_initializer(0.1), #quant_weights_1 0.2
tf.constant_initializer(0.62), #quant_activation_1 1.24
tf.constant_initializer(0.06), #quant_weights_2 0.16
tf.constant_initializer(1.15), #quant_x_1 2.05
tf.constant_initializer(0.05), #quant_weights_3 0.1
tf.constant_initializer(1.01), #quant_a_3 2.02
tf.constant_initializer(0.04), #quant_weights_4 0.08
tf.constant_initializer(1.13), #quant_x_2 2.26
tf.constant_initializer(0.03), #quant_weights_5 0.06
tf.constant_initializer(1.015), #quant_a_5 2.03
tf.constant_initializer(0.025), #quant_weights_6 0.05
tf.constant_initializer(0.03), #quant_a_6 0.06
tf.constant_initializer(0.025), #quant_weights_7 0.05
tf.constant_initializer(1.215), #quant_x_3 2.43
tf.constant_initializer(0.02), #quant_weights_8 0.04
tf.constant_initializer(0.775), #quant_a_9 1.55
tf.constant_initializer(0.0175), #quant_weights_9 0.035
tf.constant_initializer(0.775), #quant_a_10 1.55
tf.constant_initializer(0.0165), #quant_weights_10 0.033
tf.constant_initializer(0.685), #quant_x_4 1.37
tf.constant_initializer(0.0165), #quant_weights_11 0.033
tf.constant_initializer(0.565), #quant_a_11 1.13
tf.constant_initializer(0.158), #quant_weights_12 0.316
tf.constant_initializer(0.585), #quant_a_12 1.17
tf.constant_initializer(0.015), #quant_weights_13 0.03
tf.constant_initializer(0.685), #quant_x_5 1.37
tf.constant_initializer(0.045), #quant_weights_14 0.09
tf.constant_initializer(0.67), #quant_x_6 1.34
tf.constant_initializer(0.094) #quant_weights_15 0.188
]
range_init_list1 = [tf.constant_initializer(2.1), #quant_input
tf.constant_initializer(0.2), #quant_weights_1 0.2
tf.constant_initializer(1.24), #quant_activation_1 1.24
tf.constant_initializer(0.16), #quant_weights_2 0.16
tf.constant_initializer(2.05), #quant_x_1 2.05
tf.constant_initializer(0.1), #quant_weights_3 0.1
tf.constant_initializer(2.02), #quant_a_3 2.02
tf.constant_initializer(0.08), #quant_weights_4 0.08
tf.constant_initializer(2.26), #quant_x_2 2.26
tf.constant_initializer(0.06), #quant_weights_5 0.06
tf.constant_initializer(2.03), #quant_a_5 2.03
tf.constant_initializer(0.05), #quant_weights_6 0.05
tf.constant_initializer(0.06), #quant_a_6 0.06
tf.constant_initializer(0.05), #quant_weights_7 0.05
tf.constant_initializer(2.43), #quant_x_3 2.43
tf.constant_initializer(0.04), #quant_weights_8 0.04
tf.constant_initializer(1.55), #quant_a_9 1.55
tf.constant_initializer(0.035), #quant_weights_9 0.035
tf.constant_initializer(1.55), #quant_a_10 1.55
tf.constant_initializer(0.033), #quant_weights_10 0.033
tf.constant_initializer(1.37), #quant_x_4 1.37
tf.constant_initializer(0.033), #quant_weights_11 0.033
tf.constant_initializer(1.13), #quant_a_11 1.13
tf.constant_initializer(0.316), #quant_weights_12 0.316
tf.constant_initializer(1.17), #quant_a_12 1.17
tf.constant_initializer(0.03), #quant_weights_13 0.03
tf.constant_initializer(1.37), #quant_x_5 1.37
tf.constant_initializer(0.09), #quant_weights_14 0.09
tf.constant_initializer(1.34), #quant_x_6 1.34
tf.constant_initializer(0.188) #quant_weights_15 0.188
]
# vgg16 = models.QC_list_Vgg16_Cifar10(inp=cifar10_sample['x'],c_init_list= range_init_list,k=2,n_steps= 128) # c_init is intialized internally
vgg16 = models.PartQC_list_Vgg16_Cifar10(inp=cifar10_sample['x'],c_init_list= range_init_list1,k=3,n_steps= 64) # c_init is intialized internally
# vgg16 = models.lin_Vgg16_Cifar10(inp=cifar10_sample['x'])
# vgg16 = models.QC_Vgg16_Cifar10(inp=cifar10_sample['x'],c_init=tf.constant_initializer(0.1), k = 5, n_steps=2048)
# vgg16 = models.Q_Vgg16_Cifar10(inp=cifar10_sample['x'],m_init=tf.constant_initializer(0.01), k = 2)
set_determ = vgg16.set_deterministic()



# ------------------------set up the loss for training--------------------------
loss_obj = losses.map_loss(vgg16, tf.argmax(cifar10_sample['y'], axis=-1))
m_loss = loss_obj(range(len(vgg16.sto_params)))  # compute the loss based on all stochastic parameters
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100, 0.96, staircase=True)

#add all quantized parameters to the tf.summary
quant_vals = tf.get_collection('QUANT_VAL_APPROX')

# quant_range = tf.get_collection('QUANT_RANGE')
# print(quant_range)
# for qr in quant_range:
#     tf.summary.scalar(name='QUANT_RANGE',tensor=tf.convert_to_tensor(tf.reshape(qr,[])))
for qv in quant_vals:
    tf.summary.histogram(name=qv.name, values=qv)


# ------------------------set up the optimizer---------------------------
n_epochs = 100
l_rate = 1e-1
save_path = "../results/delete/Test_vgg16/QCVGG16_MAP/"

# ------------------------create the session and initialize--------------------------
sess = tf.Session(config=setup.config_tensorflow())
init = tf.global_variables_initializer()
sess.run(init)
sess.run(set_determ)  # we need a deterministic mapping for MAP training

# print(np.sqrt(sess.run(tf.nn.moments(tf.reshape(vgg16.W15.var_dist.loc, [-1 ,1]), axes=[0]))[-1]))
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
train = optimizer.minimize(m_loss, var_list= model_params)# global_step=global_step

for ep in range(n_epochs):
    # re-initialize the iterator on the dataset
    sess.run(train_init)
    while True:
        try:
            # perfrom the update-step
            _, loss_value,s = sess.run((train, m_loss,summary))

        except tf.errors.OutOfRangeError:
            break

    # evaluate the model on the training set
    sess.run(test_init)
    accuracy = evaluate_sample(vgg16, cifar10_sample, sess)
    print('Epoch %d, with loss %f, and accuracy %f' % (ep, loss_value, accuracy))

    runtime_summary.value[0].simple_value = accuracy
    runtime_summary.value[1].simple_value = loss_value
    # runtime_summary.value[2].simple_value = learn_rate
    summary_writer.add_summary(runtime_summary, ep)
    # Write to summary writer
    summary_writer.add_summary(s, ep)

    # save the trained model parameters
    vgg16.save_trainable(save_path + 'map_training_params.ckpt', sess)

