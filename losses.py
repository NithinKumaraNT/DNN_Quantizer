import tensorflow as tf
from tensorflow import distributions

class variational_loss(object):
    
    def __init__(self, model, y_ref, name='variational_loss'):
        """
        Defines the variational loss for a given model
        
        Args:
            q: probability of the labels
            y_ref: the reference labels / data
        """
        self.name       = name
        self.model      = model
        self.y_ref      = y_ref
        
    def log_prob(self, x):
        """
        calculate the log probability
        
        Args:
            x: the observed variables (data) x. its only the lable index in case of categorical data, tensorflow tensor
        """
        log_p = self.model.q.log_prob(x)
        log_p.set_shape(x.get_shape())
        return log_p
    
    def __call__(self, index):
        """
        Calculates the variational loss for the given model (priors, variational_posteriors and likelihood). 
        
        Args:
            index: list of integers, the index of the stochastic variables used to calculate the loss
            
        Returns:
            variational loss, tensorflow scalar
        """
        
        div_cost = tf.constant(0.0, dtype=tf.float32)
        
        #iterate through all parameter priors and calculate the divergence to the corresponding variational posterior
        for i in index:
            div_cost    = div_cost + tf.reduce_sum(self.model.sto_params[i].kl_distance())
            
        lp = self.log_prob(self.y_ref)
        return div_cost - tf.reduce_mean(lp)



class map_loss(object):
    
    def __init__(self, model, y_ref, name='map_loss'):
        """
        Defines the map loss for a given model
        
        Args:
            q: probability of the labels
            y_ref: the reference labels / data
        """
        self.name       = name
        self.model      = model
        self.y_ref      = y_ref
        
    def log_prob(self, x):
        """
        calculate the log probability
        
        Args:
            x: the observed variables (data) x. its only the lable index in case of categorical data, tensorflow tensor
        """
        
        log_p = self.model.q.log_prob(x)
        log_p.set_shape(x.get_shape())
        # log_p = tf.reshape(log_p,shape=x.get_shape())
        return log_p
    
    def __call__(self, index):
        """
        Calculates the variational loss for the given model (priors, variational_posteriors and likelihood). 
        
        Args:
            index: list of integers, the index of the stochastic variables used to calculate the loss
            
        Returns:
            variational loss, tensorflow scalar
        """
        with tf.name_scope("map_loss"):
        
            reg_cost = tf.constant(0.0, dtype=tf.float32)

            #iterate through all parameter priors and calculate the divergence to the corresponding variational posterior
            for i in index:
                reg_cost    = reg_cost - tf.reduce_sum(self.model.sto_params[i].prior_dist._log_unnormalized_prob(self.model.sto_params[i].var_dist.loc))

            lp = self.log_prob(self.y_ref)
            return reg_cost - tf.reduce_mean(lp)
