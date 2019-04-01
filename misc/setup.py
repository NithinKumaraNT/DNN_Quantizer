import tensorflow as tf

#defines code to setup the environment

def config_tensorflow(allow_growth=True):
	TF_config = tf.ConfigProto()
	TF_config.gpu_options.allow_growth = True
	TF_config.allow_soft_placement = True

	return TF_config
