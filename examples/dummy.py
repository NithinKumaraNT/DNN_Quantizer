from tensorflow.python.tools import inspect_checkpoint as chkp
import tensorflow as tf
import numpy as np
import bayesian_dnn.models as models
from bayesian_dnn.data import datasets
from bayesian_dnn.evaluation import evaluate_sample
import bayesian_dnn.stochastic as stochastic
from bayesian_dnn.misc import setup

# print all tensors in checkpoint file
# chkp.print_tensors_in_checkpoint_file(r'C:\Users\NNR5KOR\Thesis\ParserProject\bayesian_dnn\results\qclenet\k_2\map_training_params.ckpt', tensor_name='W1_var_loc', all_tensors=True)
#------------------------load the data and start a session---------------------------
# batch_size      = 1000
# mnist_gen       = datasets.MNIST_Set()
#
# #the training set:
# mnist_train     = mnist_gen(ds_format='cnn', key='train').shuffle(10000).batch(batch_size)
# mnist_test      = mnist_gen(ds_format='cnn', key='test').shuffle(10000).batch(batch_size)
# mnist_iterator  = tf.data.Iterator.from_structure(mnist_train.output_types, mnist_train.output_shapes)
# train_init      = mnist_iterator.make_initializer(mnist_train)
# test_init       = mnist_iterator.make_initializer(mnist_test)
# mnist_sample    = mnist_iterator.get_next()

# lenet = models.QCLeNet(inp=mnist_sample['x'], c_init=tf.constant_initializer(0.1),k=2)
saver = tf.train.Saver()
with tf.Session as sess:
    saver.restore(sess,r'C:\Users\NNR5KOR\Thesis\ParserProject\bayesian_dnn\results\Visualizaion\map_training_params.ckpt')




  # for ep in range(10):
  #     sess.run(test_init)
  #     accuracy = evaluate_sample(lenet, mnist_sample, sess)
  #     print('Epoch %d, and accuracy %f' % (ep, accuracy))

#
#
#
#
# # Create some variables.
# w1_loc = tf.get_variable("W1_var_loc", shape=[5,5,1,6])
# w1_scale = tf.get_variable("W1_var_scale", shape=[5,5,1,6])
#
# w2_loc = tf.get_variable("W2_var_loc", shape=[5, 5, 6, 16])
# w2_scale = tf.get_variable("W2_var_scale", shape=[5, 5, 6, 16])
#
# w3_loc = tf.get_variable("W3_var_loc", shape=(784,120))
# w3_scale = tf.get_variable("W3_var_scale", shape=(784,120))
#
# w4_loc = tf.get_variable("W4_var_loc", shape=(120,84))
# w4_scale = tf.get_variable("W4_var_scale", shape=(120,84))
#
# w5_loc = tf.get_variable("W5_var_loc", shape= (84,10))
# w5_scale = tf.get_variable("W5_var_scale", shape= (84,10))
#
# b1 = tf.get_variable("b1", shape=(1, 28, 28, 6))
# b2 = tf.get_variable("b2", shape=(1, 14, 14, 16))
# b3 = tf.get_variable("b3", shape=(1, 120))
# b4 = tf.get_variable("b4", shape=(1, 84))
# b5 = tf.get_variable("b5", shape=(1, 10))
#
# c_inp = tf.get_variable("c_quant_Flatten/flatten/Reshape", shape=(1))
# c_2 = tf.get_variable("c_quant_IteratorGetNext", shape=(1))
# c_3 = tf.get_variable("c_quant_Relu_2", shape=(1))
# c_4 = tf.get_variable("c_quant_Relu_3", shape=(1))
# c_5 = tf.get_variable("c_quant_add", shape=(1))
# c_6 = tf.get_variable("c_quant_add_12", shape=(1))
# c_7 = tf.get_variable("c_quant_add_24", shape=(1))
# c_8 = tf.get_variable("c_quant_add_36", shape=(1))
# c_9 = tf.get_variable("c_quant_add_48", shape=(1))
# c_10 = tf.get_variable("c_quant_max_pool", shape=(1))
#
#
# #W1_var_loc,W1_var_scale,c_quant_Flatten/flatten/Reshape,c_quant_IteratorGetNext,c_quant_Relu_2,c_quant_Relu_3, c_quant_add,c_quant_add_12
# #c_quant_add_24,c_quant_add_36,c_quant_add_48,c_quant_max_pool
# saver = tf.train.Saver()
# dict_w = {}
# dict_b = {}
# with tf.Session() as sess:
#     saver.restore(sess, r"C:\Users\NNR5KOR\Thesis\ParserProject\bayesian_dnn\results\qclenet\k_2\map_training_params.ckpt")
#     print("Model restored.")
#     # Check the values of the variables
#     dict_w["w1_loc_data"] = w1_loc.eval()
#     dict_w["w1_scale_data"] = w1_scale.eval()
#
#     dict_w["w2_loc_data"] = w2_loc.eval()
#     dict_w["w2_scale_data"] = w2_scale.eval()
#
#     dict_w["w3_loc_data"] = w3_loc.eval()
#     dict_w["w3_scale_data"] = w3_scale.eval()
#
#     dict_w["w4_loc_data"] = w4_loc.eval()
#     dict_w["w4_scale_data"] = w4_scale.eval()
#
#     dict_w["w5_loc_data"]= w5_loc.eval()
#     dict_w["w5_scale_data"] = w5_scale.eval()
#
#     dict_b["b1_data"]  = b1.eval()
#     dict_b["b2_data"]  = b2.eval()
#     dict_b["b3_data"]  = b3.eval()
#     dict_b["b4_data"]  = b4.eval()
#     dict_b["b5_data"]  = b5.eval()
#
#     c_inp_data = c_inp.eval()
#     c_2_data = c_2.eval()
#     c_3_data = c_3.eval()
#     c_4_data = c_4.eval()
#     c_5_data = c_5.eval()
#     c_6_data = c_6.eval()
#     c_7_data = c_7.eval()
#     c_8_data = c_8.eval()
#     c_9_data = c_9.eval()
#     c_10_data = c_10.eval()
#
#
# print(np.count_nonzero(dict_w<1))
#
# test_lenet = models.test_LeNet(inp=mnist_sample['x'],
#                                c_init=[c_inp_data, c_2_data, c_3_data, c_4_data, c_5_data, c_6_data,
#                                                              c_7_data,c_8_data,c_9_data,c_10_data],
#                                dict_w=dict_w,
#                                dict_b=dict_b,n_steps=5)
# set_determ  = test_lenet.set_deterministic()
# sess = tf.Session(config=setup.config_tensorflow())
# init = tf.global_variables_initializer()
# sess.run(init)
# sess.run(set_determ) #we need a deterministic mapping for MAP training
# with tf.Session() as sess:
#     sess.run(test_init)
#

    # mydata = mydata.reshape(mydata.shape[0], mydata.shape[1] * mydata.shape[2]*mydata.shape[3])
    # np.savetxt('text.csv', mydata, fmt='%.5f', delimiter=',')
