# # import the inspect_checkpoint library
# from tensorflow.python.tools import inspect_checkpoint as chkp
#
# # print all tensors in checkpoint file
# chkp.print_tensors_in_checkpoint_file(r"C:\Users\NNR5KOR\Thesis\ParserProject\bayesian_dnn\results\delete\Test_vgg16\plot\map_training_params.ckpt", tensor_name='', all_tensors=True)

import tensorflow as tf

tf.reset_default_graph()

# Create some variables.
v1 = tf.get_variable("W1_var_loc", shape=[3, 3, 3, 64])
v2 = tf.get_variable("W2_var_loc", shape=[3, 3, 64, 64])

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  saver.restore(sess, r"C:\Users\NNR5KOR\Thesis\ParserProject\bayesian_dnn\results\delete\Test_vgg16\plot\map_training_params.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
