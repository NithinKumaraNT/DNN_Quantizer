import tensorflow as tf
from tensorflow import distributions
from tensorflow.python.ops.distributions.kullback_leibler import _registered_kl

class Stochastic(object):

    def __init__(self, name, shape, prior_dist, var_dist, collection='STO'):
        """
        Implementation of a Stochastic variable that can have a prior and a variational distribution attached to it.
        
        Args:
            name: string, the name of the object
            shape: tuple, the shape of the stochastic variable
            prior_dist: callable, Constructor that returns tf.distributions.Distribution and takes the "name" as an argument
            var_dist: callable, Constructor that returns tf.distributions.Distribution and takes the "name" as an argument
            
        TODO: implement graph parents etc. that it is displayed correctly in the tensorflow graph
        """        
        self.collection     = collection
        self.deterministic  = tf.Variable(False, trainable=False)
        self.shape          = shape
        self.name           = name
        self.attach_var_dist(var_dist)
        self.attach_prior_dist(prior_dist)

    def attach_var_dist(self, dist):
        """
        Attaches a variational distribution to self. The KL-divergence between the distribution and the variational distribution has to be implemented!
            
        Args:
            dist: callable, Constructor that returns tf.distributions.Distribution and takes the "name" as an argument
        """ 
               
        #attach the variational distribution
        var_dist = dist(name=self.name + "_var")
        #check the shapes
        dist_shape = var_dist.sample().get_shape().as_list()
        if not (dist_shape == self.shape):
            raise NotImplementedError("The shape of the variational distribution does not match")#todo: change to shape missmatch error
        else:
            self.var_dist = var_dist
            for gp in self.var_dist._graph_parents:
                tf.add_to_collection(self.collection+'_VAR_PARAMS', gp)
        
    def attach_prior_dist(self, dist):
        """
        Attaches a prior distribution to self. The KL-divergence between the distribution and the variational distribution has to be implemented!
            
        Args:
            dist: callable, Constructor that returns tf.distributions.Distribution and takes the "name" as an argument
        """ 
               
        #attach the prior distribution
        prior_dist = dist(name=self.name + "_prior")
        #check the shapes
        dist_shape = prior_dist.sample().get_shape().as_list()
        if not (dist_shape == self.shape):
            raise NotImplementedError("The shape of the prior distribution does not match")#todo: change to shape missmatch error
        else:
            self.prior_dist = prior_dist
            for gp in self.prior_dist._graph_parents:
                tf.add_to_collection(self.collection+'_PRIOR_PARAMS', gp)
        
    def set_deterministic(self):
        sd = tf.assign(self.deterministic, True)
        return sd
    
    def set_stochastic(self):
        ss = tf.assign(self.deterministic, False)
        return ss
        
    def __call__(self, seed=None):
        """
        Samples from the variational distribution.
        
        Args:
            seed: the seed to generate the random numbers
        """
        with tf.name_scope("variational_dist"):
            if self.var_dist is None:
                #check wheather a variational distribution is attached
                raise NotImplementedError("No variational distribution attached to %s" % (self.name))
            samples = self.var_dist.sample() * tf.cast(tf.logical_not(self.deterministic), dtype=tf.float32) + self.var_dist.loc * tf.cast(self.deterministic, dtype=tf.float32)

            return samples
            
    def kl_distance(self):
        """
        Computes the kl distance between self.var_dist and self.prior_dist
        """
        with tf.name_scope("kl_divergence"):
            return distributions.kl_divergence(self.var_dist, self.prior_dist)
    
            
#----------------------------------TEST----------------------------------            
if __name__ == "__main__":

    from misc import setup
    import numpy as np
    import matplotlib.pyplot as plt
    
    mu_1  = tf.placeholder(tf.float32, shape=[1])
    std_1 = tf.placeholder(tf.float32, shape=[1])
    
    mu_2  = tf.placeholder(tf.float32, shape=[1])
    std_2 = tf.placeholder(tf.float32, shape=[1])
    
    prior_dist = lambda name: tf.distributions.Normal(name=name, loc=mu_1, scale=std_1)
    var_dist   = lambda name: tf.distributions.Normal(name=name, loc=mu_2, scale=std_2)
    
    sto_var = Stochastic("sto_var", [1], prior_dist, var_dist)
    
    xs  = sto_var()
    
    sess = tf.Session(config=setup.config_tensorflow())
    init = tf.global_variables_initializer()
    sess.run(init)    

    res = sess.run([xs], feed_dict={mu_1: [0], mu_2: [0], std_1: [1], std_2: [1]})
    sess.close()
    print(res)

    
