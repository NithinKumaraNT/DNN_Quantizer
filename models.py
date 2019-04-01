import numpy as np
import tensorflow as tf
from itertools import cycle
import bayesian_dnn.stochastic as stochastic
from bayesian_dnn.quantization import quant_convolution as qc
from bayesian_dnn.quantization import quant_tensordot as qt
from bayesian_dnn.quantization import quant_add as qa
from bayesian_dnn.quantization import UniformLinear as UniformLinear
from bayesian_dnn.quantization import ClippedUniformQuantizer as ClippedUniformQuantizer
from bayesian_dnn.quantization import ApproxUniformQuantizer as ApproxUniformQuantizer
from bayesian_dnn.quantization import ClippedApproxUniformQuantizer as ClippedApproxUniformQuantizer

#---------------------------helper functions-----------------------------
def get_normal_dist(name, shape, mean_initializer=tf.zeros_initializer(), scale=1.0, trainable=True):
    """
    Function that returns a normal distribution with mean initializer mean_initializer and a specific scale
    
    Args:
        name: string, the name of the distribution
        shape: tuple, the shape of the distribution
        mean_initializer: tf.initializer, how the mean of the distribution is initialized. Default is zeros_initializer.
        scale: float, the standard deviation used for each element of the normal distribution
        trainable: boolean, are the parameters loc and scale of the distribution trainable. Default is True.
    """
    
    loc     = tf.get_variable(name=name+'_loc', shape=shape, initializer=mean_initializer, trainable=trainable, dtype=tf.float32)
    scale   = tf.get_variable(name=name+'_scale', shape=shape, initializer=tf.constant_initializer(scale * np.ones(shape)), trainable=trainable, dtype=tf.float32)

    p = tf.distributions.Normal(name    = name, 
                                loc     = loc, 
                                scale   = scale)
    return p
               

#---------------------------base class for models-----------------------------
class Classification_Model(object):
    def __init__(self, inp):
        """
        Base class for models for classification. All child classes have to implement _get_logits() and _get_q()
        """
        self.inp    = inp
        self.logits = self.get_logits()
        self.q      = self.get_q(self.logits)
        if self.__class__.__name__ is 'QC_approx_Uni_LeNet':
            self.logits_uniform = self.get_logits_uniform()
            self.q_uniform = self.get_q_uniform(self.logits_uniform)

    def get_logits_uniform(self):
        return self._get_logits_uniform()

    def get_q_uniform(self, logits):
        """
        Returns the probability mass function for given logits.

        Args:
            logits: tf.tensor, the logits defined by the model
        """
        return self._get_q_uniform(logits)

    def _get_logits_uniform(self):
        """
        All Classification models have to implement this method
        """
        return None

    def _get_q_uniform(self, logits):
        """
        All Classification models have to implement this method

        Args:
            logits: tf.tensor, the logits defined by the model
        """
        return None
        #___________________________

        
    def get_logits(self):
        """
        Returns the logits for given input inp.
        """
        return self._get_logits()
    
    def get_q(self, logits):
        """
        Returns the probability mass function for given logits.
        
        Args:
            logits: tf.tensor, the logits defined by the model
        """
        return self._get_q(logits)
    
    def _get_logits(self):
        """
        All Classification models have to implement this method
        """
        return None
    
    def _get_q(self, logits):
        """
        All Classification models have to implement this method
        
        Args:
            logits: tf.tensor, the logits defined by the model
        """
        return None
    
    def save_trainable(self, path, session):
        """
        Saves all the trainable weights in a .ckpt file
        
        Args:
            path: string, the path where to save the trainable weights
            session: tf.Session, the session to save the weights from
        """
        params   = tf.trainable_variables()
        saver    = tf.train.Saver(params)
        return saver.save(session, path)
     
    def load_trainable(self, path, session):
        """
        Loads all the trainable weights from a .ckpt file
        
        Args:
            path: string, the path where to load the trainable weights from
            session: tf.Session, the session to restore the weights to
        """
        params   = tf.trainable_variables()
        saver    = tf.train.Saver(params)
        return saver.restore(session, path)
    
    def set_deterministic(self):
        """
        Sets all the stochastic parameters of the model as deterministic ones. Only
        the mean of the stochastic parameters is used to perform inference.
        """
        sd = [sp.set_deterministic() for sp in self. sto_params]
        return sd
    
    def set_stochastic(self):
        """
        Sets all the stochastic parameters of the model as stochastic. New 
        parameter realizations are drawn from the parameter distribution for inference.
        """
        ss = [sp.set_stochastic() for sp in self. sto_params]
        return ss
    

#------------------------definition of a LeNet5--------------------------

class LeNet(Classification_Model):
    
    def __init__(self, inp):
        #------------------------create the parameters--------------------------
        with tf.name_scope('Lenet'):
            with tf.name_scope('weights_layer1_init'):
                self.W1 = stochastic.Stochastic('W1',
                                        [5,5,1,6],
                                        prior_dist = lambda name: get_normal_dist(name, (5,5,1,6), scale=100.0, trainable=False),
                                        var_dist = lambda name: get_normal_dist(name, (5,5,1,6), scale=0.1, trainable=True, mean_initializer=tf.contrib.layers.xavier_initializer()))
            with tf.name_scope('weights_layer2_init'):
                self.W2 = stochastic.Stochastic('W2',
                                        [5,5,6,16],
                                        prior_dist = lambda name: get_normal_dist(name, (5,5,6,16), scale=100.0, trainable=False),
                                        var_dist = lambda name: get_normal_dist(name, (5,5,6,16), scale=0.1, trainable=True, mean_initializer=tf.contrib.layers.xavier_initializer()))
            with tf.name_scope('weights_layer3_init'):
                self.W3 = stochastic.Stochastic('W3',
                                        [784,120],
                                        prior_dist = lambda name: get_normal_dist(name, (784,120), scale=100.0, trainable=False),
                                        var_dist = lambda name: get_normal_dist(name, (784,120), scale=0.1, trainable=True, mean_initializer=tf.contrib.layers.xavier_initializer()))
            with tf.name_scope('weights_layer4_init'):
                self.W4 = stochastic.Stochastic('W4',
                                        [120,84],
                                        prior_dist = lambda name: get_normal_dist(name, (120,84), scale=100.0, trainable=False),
                                        var_dist = lambda name: get_normal_dist(name, (120,84), scale=0.1, trainable=True, mean_initializer=tf.contrib.layers.xavier_initializer()))
            with tf.name_scope('weights_layer5_init'):
                self.W5 = stochastic.Stochastic('W5',
                                        [84,10],
                                        prior_dist = lambda name: get_normal_dist(name, (84,10), scale=100.0, trainable=False),
                                        var_dist = lambda name: get_normal_dist(name, (84,10), scale=0.1, trainable=True, mean_initializer=tf.contrib.layers.xavier_initializer()))

            self.sto_params = [self.W1, self.W2, self.W3, self.W4, self.W5]
            with tf.name_scope('bias_layer1_init'):
                self.b1 = tf.get_variable(name='b1', shape=(1,28,28,6), dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
            with tf.name_scope('bias_layer2_init'):
                self.b2 = tf.get_variable(name='b2', shape=(1,14,14,16), dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
            with tf.name_scope('bias_layer3_init'):
                self.b3 = tf.get_variable(name='b3', shape=(1,120), dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
            with tf.name_scope('bias_layer4_init'):
                self.b4 = tf.get_variable(name='b4', shape=(1,84), dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
            with tf.name_scope('bias_layer5_init'):
                self.b5 = tf.get_variable(name='b5', shape=(1,10), dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)

            self.det_params = [self.b1, self.b2, self.b3, self.b4, self.b5]
            for d in self.det_params:
                tf.add_to_collection('DET_PARAMS', d)

            super(LeNet, self).__init__(inp=inp)
        
    def _get_logits(self):

        #------------------------create the network graph--------------------------
        #layer 1:
        a_1 = tf.nn.convolution(self.inp, self.W1(), padding="SAME") + self.b1
        x_1 = tf.nn.pool(tf.nn.relu(a_1), [2,2], "MAX","SAME", strides=[2,2])

        #layer 2:
        a_2 = tf.nn.convolution(x_1, self.W2(), padding="SAME") + self.b2
        x_2 = tf.contrib.layers.flatten(tf.nn.pool(tf.nn.relu(a_2), [2,2], "MAX","SAME", strides=[2,2]))

        #layer 3:
        a_3 = tf.tensordot(x_2, self.W3(), axes=1) + self.b3
        x_3 = tf.nn.relu(a_3)
            
        #layer 4:
        a_4 = tf.tensordot(x_3, self.W4(), axes=1) + self.b4
        x_4 = tf.nn.relu(a_4)

        #layer 5 returning the logits:
        a_5 = tf.tensordot(x_4, self.W5(), axes=1) + self.b5
        return a_5
                
    def _get_q(self, logits):
        return tf.distributions.Categorical(name='q', logits=logits)

    def _get_q_uniform(self, logits):
        return tf.distributions.Categorical(name='q', logits=logits)

#------------------------definition of multiple quantized LeNet5--------------------------
#quantized LeNet without clipping
class QLeNet(LeNet):
    
    def __init__(self, inp, m_init, k):
        """
        Instantiates a LeNet with quantized parameters and layer inputs. The quantizer does not
        clip to a fixed range.
        
        Args:
            inp: tf.tensor, the input of the network
            m_init: tf.initializer, the initializer for the resolution of the quantizers
            k: integer, the approximation order for approximate quantization
        """
        self.m_init = m_init
        self.k = k
        super(QLeNet, self).__init__(inp)
        
    def linear(self, name):
        return UniformLinear(name=name)
    
    def quant(self, name):
        return ApproxUniformQuantizer(m_init=self.m_init, k=self.k, name=name)
        
    def _get_logits(self):
        # ------------------------create the network graph--------------------------
        # layer 1:
        with tf.name_scope('layer1'):
            a_1 = qc(self.inp, self.W1(), self.quant(name="quant_input"), self.quant(name="quant_weights_1"),
                     padding="SAME") + self.b1
            x_1 = tf.nn.pool(tf.nn.relu(a_1), [2, 2], "MAX", "SAME", strides=[2, 2])

        # layer 2:
        with tf.name_scope('layer2'):
            a_2 = qc(x_1, self.W2(), self.quant(name="quant_activation_1"), self.quant(name="quant_weights_2"),
                     padding="SAME") + self.b2
            x_2 = tf.contrib.layers.flatten(tf.nn.pool(tf.nn.relu(a_2), [2, 2], "MAX", "SAME", strides=[2, 2]))

        # layer 3:
        with tf.name_scope('layer3'):
            a_3 = qt(x_2, self.W3(), self.quant(name="quant_activation_2"), self.quant(name="quant_weights_3"),
                     axes=1) + self.b3
            x_3 = tf.nn.relu(a_3)

        # layer 4:
        with tf.name_scope('layer4'):
            a_4 = qt(x_3, self.W4(), self.quant(name="quant_activation_3"), self.quant(name="quant_weights_4"),
                     axes=1) + self.b4
            x_4 = tf.nn.relu(a_4)

        # layer 5 returning the logits:
        with tf.name_scope('output_layer'):
            a_5 = qt(x_4, self.W5(), self.quant(name="quant_activation_4"), self.quant(name="quant_weights_5"),
                     axes=1) + self.b5
            return a_5

# Uniform quantized LeNet with clipping
class QHardLeNet(LeNet):

    def __init__(self, inp,c_init):
        """
        Instantiates a LeNet with quantized parameters and layer inputs. The quantizer does not
        clip to a fixed range.

        Args:
            inp: tf.tensor, the input of the network
            m_init: tf.initializer, the initializer for the resolution of the quantizers
            k: integer, the approximation order for approximate quantization
        """
        self.c_init = c_init
        super(QHardLeNet, self).__init__(inp)

    def linear(self, name):
        return UniformLinear(name=name)

    def quant(self, name):
        return ClippedUniformQuantizer(c_init=self.c_init,name=name)

    def _get_logits(self):

        # ------------------------create the network graph--------------------------
        # layer 1:
        with tf.name_scope('layer1'):
            a_1 = qc(self.inp, self.W1(), self.quant(name="quant_input"), self.quant(name="quant_weights_1"), padding="SAME")+ self.b1
            x_1 = tf.nn.pool(tf.nn.relu(a_1), [2, 2], "MAX", "SAME", strides=[2, 2])

        # layer 2:
        with tf.name_scope('layer2'):
            a_2 = qc(x_1, self.W2(), self.quant(name="quant_activation_1"), self.quant(name="quant_weights_2"), padding="SAME")+ self.b2
            x_2 = tf.contrib.layers.flatten(tf.nn.pool(tf.nn.relu(a_2), [2, 2], "MAX", "SAME", strides=[2, 2]))

        # layer 3:
        with tf.name_scope('layer3'):
            a_3 = qt(x_2, self.W3(), self.quant(name="quant_activation_2"), self.quant(name="quant_weights_3"), axes=1)+ self.b3
            x_3 = tf.nn.relu(a_3)

        # layer 4:
        with tf.name_scope('layer4'):
            a_4 = qt(x_3, self.W4(), self.quant(name="quant_activation_3"), self.quant(name="quant_weights_4"), axes=1)+ self.b4
            x_4 = tf.nn.relu(a_4)

        # layer 5 returning the logits:
        with tf.name_scope('output_layer'):
            a_5 = qt(x_4, self.W5(), self.quant(name="quant_activation_4"), self.quant(name="quant_weights_5"), axes=1)+ self.b5
            return a_5

# Approx quantized LeNet with clipping
class QCLeNet(LeNet):

    def __init__(self, inp,c_init, k, n_steps=5):
        """
        Instantiates a LeNet with quantized parameters and layer inputs. The quantizer does not
        clip to a fixed range.

        Args:
            inp: tf.tensor, the input of the network
            m_init: tf.initializer, the initializer for the resolution of the quantizers
            k: integer, the approximation order for approximate quantization
        """
        self.c_init = c_init
        self.k = k
        self.n_steps = n_steps
        super(QCLeNet, self).__init__(inp)

    def linear(self, name):
        return UniformLinear(name=name)

    def quant(self, name):
        return ClippedApproxUniformQuantizer(c_init=self.c_init, k=self.k, n_steps=self.n_steps, name=name)

    def _get_logits(self):

        # ------------------------create the network graph--------------------------
        # layer 1:
        with tf.name_scope('layer1'):
            a_1 = qc(self.inp, self.W1(), self.quant(name="quant_input"), self.quant(name="quant_weights_1"), padding="SAME")+ self.b1
            x_1 = tf.nn.pool(tf.nn.relu(a_1), [2, 2], "MAX", "SAME", strides=[2, 2])

        # layer 2:
        with tf.name_scope('layer2'):
            a_2 = qc(x_1, self.W2(), self.quant(name="quant_activation_1"), self.quant(name="quant_weights_2"), padding="SAME")+ self.b2
            x_2 = tf.contrib.layers.flatten(tf.nn.pool(tf.nn.relu(a_2), [2, 2], "MAX", "SAME", strides=[2, 2]))

        # layer 3:
        with tf.name_scope('layer3'):
            a_3 = qt(x_2, self.W3(), self.quant(name="quant_activation_2"), self.quant(name="quant_weights_3"), axes=1)+ self.b3
            x_3 = tf.nn.relu(a_3)

        # layer 4:
        with tf.name_scope('layer4'):
            a_4 = qt(x_3, self.W4(), self.quant(name="quant_activation_3"), self.quant(name="quant_weights_4"), axes=1)+ self.b4
            x_4 = tf.nn.relu(a_4)

        # layer 5 returning the logits:
        with tf.name_scope('output_layer'):
            a_5 = qt(x_4, self.W5(), self.quant(name="quant_activation_4"), self.quant(name="quant_weights_5"), axes=1)+ self.b5
            return a_5

#quantized LeNet with clipping both approx and uniform
class QC_approx_Uni_LeNet(LeNet):
    
    def __init__(self, inp, c_init_list, k, n_steps):
        """
        Instantiates a LeNet with quantized parameters and layer inputs. The quantizer does 
        clip to a fixed range c.
        
        Args:
            inp: tf.tensor, the input of the network
            c_init: tf.initializer, the initializer for the range c of the quantizers with clipping
            k: integer), the approximation order for approximate quantization
            n_steps: positive odd integer (numpy), the number of quantization steps used for training
        """
        self.c_init     = cycle(c_init_list)
        self.k          = k
        self.n_steps    = n_steps
        super(QC_approx_Uni_LeNet, self).__init__(inp)
        
    def linear(self, name):
        return UniformLinear(name=name)
    
    def quant(self, name):
        with tf.variable_scope("Quant_vars"):
            return ClippedApproxUniformQuantizer(c_init=next(self.c_init), k=self.k, n_steps=self.n_steps, name=name)

    def quant_uni(self, name):
        with tf.variable_scope("Quant_vars",reuse=True):
            return ClippedUniformQuantizer(c_init=next(self.c_init),n_steps=self.n_steps, name=name)


    def _get_logits(self):

        # ------------------------create the network graph--------------------------
        # layer 1:
        with tf.name_scope('layer1'):
            a_1 = qc(self.inp, self.W1(), self.quant(name="quant_input"), self.quant(name="quant_weights_1"), padding="SAME")+ self.b1
            x_1 = tf.nn.pool(tf.nn.relu(a_1), [2, 2], "MAX", "SAME", strides=[2, 2])

        # layer 2:
        with tf.name_scope('layer2'):
            a_2 = qc(x_1, self.W2(), self.quant(name="quant_activation_1"), self.quant(name="quant_weights_2"), padding="SAME")+ self.b2
            x_2 = tf.contrib.layers.flatten(tf.nn.pool(tf.nn.relu(a_2), [2, 2], "MAX", "SAME", strides=[2, 2]))

        # layer 3:
        with tf.name_scope('layer3'):
            a_3 = qt(x_2, self.W3(), self.quant(name="quant_activation_2"), self.quant(name="quant_weights_3"), axes=1)+ self.b3
            x_3 = tf.nn.relu(a_3)

        # layer 4:
        with tf.name_scope('layer4'):
            a_4 = qt(x_3, self.W4(), self.quant(name="quant_activation_3"), self.quant(name="quant_weights_4"), axes=1)+ self.b4
            x_4 = tf.nn.relu(a_4)

        # layer 5 returning the logits:
        with tf.name_scope('output_layer'):
            a_5 = qt(x_4, self.W5(), self.quant(name="quant_activation_4"), self.quant(name="quant_weights_5"), axes=1)+ self.b5
            return a_5

    def _get_logits_uniform(self):
        # ------------------------create the network graph--------------------------
        # layer 1:
        with tf.name_scope('layer1'):
            a_1 = qc(self.inp, self.W1(), self.quant_uni(name="quant_input"), self.quant_uni(name="quant_weights_1"),
                     padding="SAME") + self.b1
            x_1 = tf.nn.pool(tf.nn.relu(a_1), [2, 2], "MAX", "SAME", strides=[2, 2])

        # layer 2:
        with tf.name_scope('layer2'):
            a_2 = qc(x_1, self.W2(), self.quant_uni(name="quant_activation_1"), self.quant_uni(name="quant_weights_2"),
                     padding="SAME") + self.b2
            x_2 = tf.contrib.layers.flatten(tf.nn.pool(tf.nn.relu(a_2), [2, 2], "MAX", "SAME", strides=[2, 2]))

        # layer 3:
        with tf.name_scope('layer3'):
            a_3 = qt(x_2, self.W3(), self.quant_uni(name="quant_activation_2"), self.quant_uni(name="quant_weights_3"),
                     axes=1) + self.b3
            x_3 = tf.nn.relu(a_3)

        # layer 4:
        with tf.name_scope('layer4'):
            a_4 = qt(x_3, self.W4(), self.quant_uni(name="quant_activation_3"), self.quant_uni(name="quant_weights_4"),
                     axes=1) + self.b4
            x_4 = tf.nn.relu(a_4)

        # layer 5 returning the logits:
        with tf.name_scope('output_layer'):
            a_5 = qt(x_4, self.W5(), self.quant_uni(name="quant_activation_4"), self.quant_uni(name="quant_weights_5"),
                     axes=1) + self.b5
            return a_5

# Approx quantized LeNet with clipping
class QCLeNet_list_init(LeNet):

    def __init__(self, inp, c_init_list, k, n_steps=5):
        """
        Instantiates a LeNet with quantized parameters and layer inputs. The quantizer does not
        clip to a fixed range.

        Args:
            inp: tf.tensor, the input of the network
            m_init: tf.initializer, the initializer for the resolution of the quantizers
            k: integer, the approximation order for approximate quantization
        """
        self.c_init = cycle(c_init_list)
        self.k = k
        self.n_steps = n_steps
        super(QCLeNet_list_init, self).__init__(inp)

    def linear(self, name):
        return UniformLinear(name=name)

    def quant(self, name):
        return ClippedApproxUniformQuantizer(c_init=next(self.c_init_list), k=self.k, n_steps=self.n_steps, name=name)

    def xavier_init(self):
        return tf.contrib.layers.xavier_initializer(uniform=True,seed=None,dtype=tf.float32)

    def _get_logits(self):

        # ------------------------create the network graph--------------------------
        # layer 1:
        with tf.name_scope('layer1'):
            a_1 = qc(self.inp, self.W1(), self.quant(c_init=self.xavier_init(),name="quant_input"),
                     self.quant(c_init=self.xavier_init(),name="quant_weights_1"), padding="SAME")+ self.b1
            x_1 = tf.nn.pool(tf.nn.relu(a_1), [2, 2], "MAX", "SAME", strides=[2, 2])

        # layer 2:
        with tf.name_scope('layer2'):
            a_2 = qc(x_1, self.W2(), self.quant(c_init=self.xavier_init(),name="quant_activation_1"),
                     self.quant(c_init=self.xavier_init(),name="quant_weights_2"), padding="SAME")+ self.b2
            x_2 = tf.contrib.layers.flatten(tf.nn.pool(tf.nn.relu(a_2), [2, 2], "MAX", "SAME", strides=[2, 2]))

        # layer 3:
        with tf.name_scope('layer3'):
            a_3 = qt(x_2, self.W3(), self.quant(c_init=self.xavier_init(),name="quant_activation_2"),
                     self.quant(c_init=self.xavier_init(),name="quant_weights_3"), axes=1)+ self.b3
            x_3 = tf.nn.relu(a_3)

        # layer 4:
        with tf.name_scope('layer4'):
            a_4 = qt(x_3, self.W4(), self.quant(c_init=self.xavier_init(),name="quant_activation_3"),
                     self.quant(c_init=self.xavier_init(),name="quant_weights_4"), axes=1)+ self.b4
            x_4 = tf.nn.relu(a_4)

        # layer 5 returning the logits:
        with tf.name_scope('output_layer'):
            a_5 = qt(x_4, self.W5(), self.quant(c_init=self.xavier_init(),name="quant_activation_4"),
                     self.quant(c_init=self.xavier_init(),name="quant_weights_5"), axes=1)+ self.b5
            return a_5

# Approx quantized LeNet with clipping
class QCLeNet_list_init_try1(LeNet):

    def __init__(self, inp, c_init_list, k, n_steps=100):
        """
        Instantiates a LeNet with quantized parameters and layer inputs. The quantizer does not
        clip to a fixed range.

        Args:
            inp: tf.tensor, the input of the network
            m_init: tf.initializer, the initializer for the resolution of the quantizers
            k: integer, the approximation order for approximate quantization
        """
        self.c_init_list = cycle(c_init_list)
        self.k = k
        self.n_steps = n_steps
        super(QCLeNet_list_init_try1, self).__init__(inp)

    def linear(self, name):
        return UniformLinear(name=name)

    def quant(self,name):
        return ClippedApproxUniformQuantizer(c_init=next(self.c_init_list), n_steps=self.n_steps, name=name)

    def _get_logits(self):
        # ------------------------create the network graph--------------------------
        # layer 1:
        with tf.name_scope('layer1'):
            a_1 = qc(self.inp, self.W1(), self.quant(name="quant_input"), self.quant(name="quant_weights_1"),
                     padding="SAME") + self.b1
            x_1 = tf.nn.pool(tf.nn.relu(a_1), [2, 2], "MAX", "SAME", strides=[2, 2])

        # layer 2:
        with tf.name_scope('layer2'):
            a_2 = qc(x_1, self.W2(), self.quant(name="quant_activation_1"), self.quant(name="quant_weights_2"),
                     padding="SAME") + self.b2
            x_2 = tf.contrib.layers.flatten(tf.nn.pool(tf.nn.relu(a_2), [2, 2], "MAX", "SAME", strides=[2, 2]))

        # layer 3:
        with tf.name_scope('layer3'):
            a_3 = qt(x_2, self.W3(), self.quant(name="quant_activation_2"), self.quant(name="quant_weights_3"),
                     axes=1) + self.b3
            x_3 = tf.nn.relu(a_3)

        # layer 4:
        with tf.name_scope('layer4'):
            a_4 = qt(x_3, self.W4(), self.quant(name="quant_activation_3"), self.quant(name="quant_weights_4"),
                     axes=1) + self.b4
            x_4 = tf.nn.relu(a_4)

        # layer 5 returning the logits:
        with tf.name_scope('output_layer'):
            a_5 = qt(x_4, self.W5(), self.quant(name="quant_activation_4"), self.quant(name="quant_weights_5"),
                     axes=1) + self.b5
            return a_5

# ------------------------definition of a SVHN--------------------------

class SVHN(Classification_Model):

    def __init__(self, inp):

        # ------------------------create the parameters--------------------------
        self.W1 = stochastic.Stochastic('W1',
                                        [5, 5, 3, 48],
                                        prior_dist=lambda name: get_normal_dist(name, (5, 5, 3, 48), scale=100.0,
                                                                                trainable=False),
                                        var_dist=lambda name: get_normal_dist(name, (5, 5, 3, 48), scale=0.1,
                                                                              trainable=True,
                                                                              mean_initializer=tf.contrib.layers.xavier_initializer())) #conv_1

        self.W2 = stochastic.Stochastic('W2',
                                        [5, 5, 48, 64],
                                        prior_dist=lambda name: get_normal_dist(name, (5, 5, 48, 64), scale=100.0,
                                                                                trainable=False),
                                        var_dist=lambda name: get_normal_dist(name, (5, 5, 48, 64), scale=0.1,
                                                                              trainable=True,
                                                                              mean_initializer=tf.contrib.layers.xavier_initializer())) #conv_2

        self.W3 = stochastic.Stochastic('W3',
                                        [5, 5, 64, 128],
                                        prior_dist=lambda name: get_normal_dist(name, (5, 5, 64, 128), scale=100.0,
                                                                                trainable=False),
                                        var_dist=lambda name: get_normal_dist(name, (5, 5, 64, 128), scale=0.1,
                                                                              trainable=True,
                                                                              mean_initializer=tf.contrib.layers.xavier_initializer())) #conv_3

        self.W4 = stochastic.Stochastic('W4',
                                        [2048, 256],
                                        prior_dist=lambda name: get_normal_dist(name, (2048, 256), scale=100.0,
                                                                                trainable=False),
                                        var_dist=lambda name: get_normal_dist(name, (2048, 256), scale=0.1,
                                                                              trainable=True,
                                                                              mean_initializer=tf.contrib.layers.xavier_initializer())) #FC1

        self.W5 = stochastic.Stochastic('W5',
                                        [256, 128],
                                        prior_dist=lambda name: get_normal_dist(name, (256, 128), scale=100.0,
                                                                                trainable=False),
                                        var_dist=lambda name: get_normal_dist(name, (256, 128), scale=0.1, trainable=True,
                                                                              mean_initializer=tf.contrib.layers.xavier_initializer())) #FC2
        self.W6 = stochastic.Stochastic('W6',
                                        [128, 10],
                                        prior_dist=lambda name: get_normal_dist(name, (128, 10), scale=100.0,
                                                                                trainable=False),
                                        var_dist=lambda name: get_normal_dist(name, (128, 10), scale=0.1,
                                                                              trainable=True,
                                                                              mean_initializer=tf.contrib.layers.xavier_initializer())) #output layer

        self.sto_params = [self.W1, self.W2, self.W3, self.W4, self.W5, self.W6]

        self.b1 = tf.get_variable(name='b1', shape=(1, 32, 32, 48), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.b2 = tf.get_variable(name='b2', shape=(1, 16, 16, 64), dtype=tf.float32,
                                  initializer=tf.zeros_initializer(), trainable=True)
        self.b3 = tf.get_variable(name='b3', shape=(1, 8, 8, 128), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.b4 = tf.get_variable(name='b4', shape=(1, 256), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.b5 = tf.get_variable(name='b5', shape=(1, 128), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True)
        self.b6 = tf.get_variable(name='b6', shape=(1, 10), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True)

        self.det_params = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6]
        for d in self.det_params:
            tf.add_to_collection('DET_PARAMS', d)

        super(SVHN, self).__init__(inp=inp)

    def _get_logits(self):
        # ------------------------create the network graph--------------------------
        # Conv layer 1:
        a_1 = tf.nn.conv2d(self.inp, self.W1(),strides=[1, 1, 1, 1], padding="SAME") + self.b1
        x_1 = tf.nn.max_pool(tf.nn.relu(a_1), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 2:
        a_2 = tf.nn.conv2d(x_1, self.W2(),[1, 1, 1, 1], padding="SAME") + self.b2
        x_2 = tf.nn.max_pool(tf.nn.relu(a_2), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 3:
        a_3 = tf.nn.conv2d(x_2, self.W3(), [1, 1, 1, 1], padding="SAME") + self.b3
        x_3 =tf.contrib.layers.flatten(tf.nn.max_pool(tf.nn.relu(a_3), [1, 2, 2, 1], [1, 2, 2, 1], "SAME"))

        # fc layer 4:
        a_4 = tf.tensordot(x_3, self.W4(), axes=1) + self.b4
        x_4 = tf.nn.relu(a_4)

        # fc layer 5 :
        a_5 = tf.tensordot(x_4, self.W5(), axes=1) + self.b5
        x_5 = tf.nn.relu(a_5)

        # output layer 6 returning the logits:
        a_6 = tf.tensordot(x_5, self.W6(), axes=1) + self.b6
        return a_6

    def _get_q(self, logits):
        return tf.distributions.Categorical(name='q', logits=logits)

class Vgg16_Cifar10(Classification_Model):

    def add_layer(self,name,param_dim):
        return stochastic.Stochastic(name,
                                     param_dim,
                              prior_dist=lambda name: get_normal_dist(name, param_dim, scale=100.0,
                                                                      trainable=False),
                              var_dist=lambda name: get_normal_dist(name, param_dim, scale=0.1,
                                                                    trainable=True,
                                                                    mean_initializer=tf.contrib.layers.xavier_initializer()))


    def __init__(self, inp):


        # ------------------------create the parameters--------------------------
        self.W1 = self.add_layer('W1',[3, 3, 3, 64]) #conv1
        self.W2 = self.add_layer('W2',[3, 3, 64, 64]) #conv2
        # max pool
        self.W3 = self.add_layer('W3', [3, 3, 64, 128])  # conv3
        self.W4 = self.add_layer('W4', [3, 3, 128, 128])  # conv4
        # max pool
        self.W5 = self.add_layer('W5', [3, 3, 128, 256])  # conv5
        self.W6 = self.add_layer('W6', [3, 3, 256, 256])  # conv6
        self.W7 = self.add_layer('W7', [3, 3, 256, 256])  # conv7
        # max pool
        self.W8 = self.add_layer('W8', [3, 3, 256, 512])  # conv8
        self.W9 = self.add_layer('W9', [3, 3, 512, 512])  # conv9
        self.W10 = self.add_layer('W10', [3, 3, 512, 512])  # conv10
        # max pool
        self.W11 = self.add_layer('W11', [3, 3, 512, 512])  # conv11
        self.W12 = self.add_layer('W12', [3, 3, 512, 512])  # conv12
        self.W13 = self.add_layer('W13', [3, 3, 512, 512])  # conv13
        #max pool
        self.W14 = self.add_layer('W14', [512, 1024])  # conv14 ( [None,1,1,512] = 1x1x512 = 512 neurons )
        self.W15 = self.add_layer('W15', [1024, 512])  # conv15
        self.W16 = self.add_layer('W16', [512, 10])   # conv output layer


        self.sto_params = [self.W1, self.W2, self.W3, self.W4, self.W5, self.W6, self.W7,self.W8,
                           self.W9, self.W10, self.W11, self.W12, self.W13, self.W14, self.W15, self.W16]

        self.b1 = tf.get_variable(name='b1', shape=(1, 32, 32, 64), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True)  #conv_1
        self.b2 = tf.get_variable(name='b2', shape=(1, 32, 32, 64), dtype=tf.float32,
                                  initializer=tf.zeros_initializer(),
                                  trainable=True) #conv_2
        #max pool
        self.b3 = tf.get_variable(name='b3', shape=(1, 16, 16, 128), dtype=tf.float32,
                                  initializer=tf.zeros_initializer(), trainable=True)  #conv_3
        self.b4 = tf.get_variable(name='b4', shape=(1, 16, 16, 128), dtype=tf.float32,
                                  initializer=tf.zeros_initializer(), trainable=True)  #conv_4
        # max pool

        self.b5 = tf.get_variable(name='b5', shape=(1, 8, 8, 256), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True) #conv_5
        self.b6 = tf.get_variable(name='b6', shape=(1, 8, 8, 256), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True) #conv_6
        self.b7 = tf.get_variable(name='b7', shape=(1, 8, 8, 256), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True) #conv_7
        # max pool

        self.b8 = tf.get_variable(name='b8', shape=(1, 4, 4, 512), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True) #conv_8
        self.b9 = tf.get_variable(name='b9', shape=(1, 4, 4, 512), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True) #conv_9
        self.b10 = tf.get_variable(name='b10', shape=(1, 4, 4, 512), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True) #conv_10
        # max pool

        self.b11 = tf.get_variable(name='b11', shape=(1, 2, 2, 512), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True) #conv_11
        self.b12 = tf.get_variable(name='b12', shape=(1, 2, 2, 512), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True) #conv_12
        self.b13 = tf.get_variable(name='b13', shape=(1, 2, 2, 512), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True) #conv_13
        
        self.b14 = tf.get_variable(name='b14', shape=(1, 1024), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True) #FC_1
        self.b15 = tf.get_variable(name='b15', shape=(1, 512), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                  trainable=True) #FC_2
        self.b16 = tf.get_variable(name='b16', shape=(1, 10), dtype=tf.float32, initializer=tf.zeros_initializer(),
                                   trainable=True)  # output

        self.det_params = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.b8, self.b9,
                           self.b10, self.b11, self.b12, self.b13, self.b14, self.b15, self.b16]
        for d in self.det_params:
            tf.add_to_collection('DET_PARAMS', d)

        super(Vgg16_Cifar10, self).__init__(inp=inp)

    def _get_logits(self):
        # ------------------------create the network graph--------------------------
        # Conv layer 1:
        a_1 = tf.nn.conv2d(self.inp, self.W1(),strides=[1, 1, 1, 1], padding="SAME") + self.b1

        # a_1 = tf.layers.batch_normalization(a_1, training=self.train_phase)
        # Conv layer 2:
        a_2 = tf.nn.conv2d(a_1, self.W2(), strides=[1, 1, 1, 1], padding="SAME") + self.b2
        # max pooling 1: 16 x 16
        x_1 = tf.nn.max_pool(tf.nn.relu(a_2), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 3:
        a_3 = tf.nn.conv2d(x_1, self.W3(), strides=[1, 1, 1, 1], padding="SAME") + self.b3
        # Conv layer 4:
        a_4 = tf.nn.conv2d(a_3, self.W4(), strides=[1, 1, 1, 1], padding="SAME") + self.b4
        # max pooling 2: 8 x 8
        x_2 = tf.nn.max_pool(tf.nn.relu(a_4), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 5:
        a_5 = tf.nn.conv2d(x_2, self.W5(), strides=[1, 1, 1, 1], padding="SAME") + self.b5
        # Conv layer 6:
        a_6 = tf.nn.conv2d(a_5, self.W6(), strides=[1, 1, 1, 1], padding="SAME") + self.b6
        # Conv layer 7:
        a_7 = tf.nn.conv2d(a_6, self.W7(), strides=[1, 1, 1, 1], padding="SAME") + self.b7
        # max pooling 3: 4 x 4
        x_3 = tf.nn.max_pool(tf.nn.relu(a_7), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 8:
        a_8 = tf.nn.conv2d(x_3, self.W8(), strides=[1, 1, 1, 1], padding="SAME") + self.b8
        # Conv layer 9:
        a_9 = tf.nn.conv2d(a_8, self.W9(), strides=[1, 1, 1, 1], padding="SAME") + self.b9
        # Conv layer 10:
        a_10 = tf.nn.conv2d(a_9, self.W10(), strides=[1, 1, 1, 1], padding="SAME") + self.b10
        # max pooling 4: 2 x 2
        x_4 = tf.nn.max_pool(tf.nn.relu(a_10), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 11:
        a_11 = tf.nn.conv2d(x_4, self.W11(), strides=[1, 1, 1, 1], padding="SAME") + self.b11
        # Conv layer 12:
        a_12 = tf.nn.conv2d(a_11, self.W12(), strides=[1, 1, 1, 1], padding="SAME") + self.b12
        # Conv layer 13:
        a_13 = tf.nn.conv2d(a_12, self.W13(), strides=[1, 1, 1, 1], padding="SAME") + self.b13
        # max pooling 5: 1 x 1
        x_5 =tf.contrib.layers.flatten(tf.nn.max_pool(tf.nn.relu(a_13), [1, 2, 2, 1], [1, 2, 2, 1], "SAME"))

        # fc layer 14:
        a_14 = tf.tensordot(x_5, self.W14(), axes=1) + self.b14
        x_6 = tf.nn.relu(a_14)

        # output layer 15 returning the logits:
        a_15 = tf.tensordot(x_6, self.W15(), axes=1) + self.b15
        x_7 = tf.nn.relu(a_15)

        a_16 = tf.tensordot(x_7, self.W16(), axes=1) + self.b16

        return a_16

    def _get_q(self, logits):
        return tf.distributions.Categorical(name='q', logits=logits)

class QC_Vgg16_Cifar10(Vgg16_Cifar10):

    def __init__(self, inp,c_init, k, n_steps=5):
        """
        Instantiates a LeNet with quantized parameters and layer inputs. The quantizer does not
        clip to a fixed range.

        Args:
            inp: tf.tensor, the input of the network
            m_init: tf.initializer, the initializer for the resolution of the quantizers
            k: integer, the approximation order for approximate quantization
        """
        self.c_init = c_init
        self.k = k
        self.n_steps = n_steps
        super(QC_Vgg16_Cifar10, self).__init__(inp)

    def linear(self, name):
        return UniformLinear(name=name)

    def quant(self, name):
        return ClippedApproxUniformQuantizer(c_init=self.c_init, k=self.k, n_steps=self.n_steps, name=name)

    def _get_logits(self):
        # ------------------------create the network graph--------------------------
        # Conv layer 1:
        a_1 = qc(self.inp, self.W1(),self.quant(name="quant_input"), self.quant(name="quant_weights_1"), padding="SAME") + self.b1

        # Conv layer 2:
        a_2 = qc(a_1, self.W2(), self.quant(name="quant_a_2"), self.quant(name="quant_weights_2"), padding="SAME") + self.b2
        # max pooling 1: 16 x 16
        x_1 = tf.nn.max_pool(tf.nn.relu(a_2), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 3:
        a_3 = qc(x_1, self.W3(), self.quant(name="quant_x_1"), self.quant(name="quant_weights_3"), padding="SAME") + self.b3
        # Conv layer 4:
        a_4 = qc(a_3, self.W4(), self.quant(name="quant_a_4"), self.quant(name="quant_weights_4"), padding="SAME") + self.b4
        # max pooling 2: 8 x 8
        x_2 = tf.nn.max_pool(tf.nn.relu(a_4), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 5:
        a_5 = qc(x_2, self.W5(), self.quant(name="quant_x_2"), self.quant(name="quant_weights_5"), padding="SAME") + self.b5
        # Conv layer 6:
        a_6 = qc(a_5, self.W6(),self.quant(name="quant_a_5"), self.quant(name="quant_weights_6"), padding="SAME") + self.b6
        # Conv layer 7:
        a_7 = qc(a_6, self.W7(), self.quant(name="quant_a_6"), self.quant(name="quant_weights_7"), padding="SAME") + self.b7
        # max pooling 3: 4 x 4
        x_3 = tf.nn.max_pool(tf.nn.relu(a_7), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 8:
        a_8 = qc(x_3, self.W8(), self.quant(name="quant_x_3"), self.quant(name="quant_weights_8"), padding="SAME") + self.b8
        # Conv layer 9:
        a_9 = qc(a_8, self.W9(), self.quant(name="quant_a_9"), self.quant(name="quant_weights_9"), padding="SAME") + self.b9
        # Conv layer 10:
        a_10 = qc(a_9, self.W10(), self.quant(name="quant_a_10"), self.quant(name="quant_weights_10"), padding="SAME") + self.b10
        # max pooling 4: 2 x 2
        x_4 = tf.nn.max_pool(tf.nn.relu(a_10), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 11:
        a_11 = qc(x_4, self.W11(), self.quant(name="quant_x_4"), self.quant(name="quant_weights_11"), padding="SAME") + self.b11
        # Conv layer 12:
        a_12 = qc(a_11, self.W12(), self.quant(name="quant_a_11"), self.quant(name="quant_weights_12"), padding="SAME") + self.b12
        # Conv layer 13:
        a_13 = qc(a_12, self.W13(), self.quant(name="quant_a_12"), self.quant(name="quant_weights_13"), padding="SAME") + self.b13
        # max pooling 5: 1 x 1
        x_5 = tf.contrib.layers.flatten(tf.nn.max_pool(tf.nn.relu(a_13), [1, 2, 2, 1], [1, 2, 2, 1], "SAME"))

        # fc layer 14:
        a_14 = qt(x_5, self.W14(),self.quant(name="quant_x_5"), self.quant(name="quant_weights_14"), axes=1) + self.b14
        x_6 = tf.nn.relu(a_14)

        # output layer 15 returning the logits:
        a_15 = qt(x_6, self.W15(),self.quant(name="quant_x_6"), self.quant(name="quant_weights_15"), axes=1) + self.b15
        x_7 = tf.nn.relu(a_15)

        a_16 = qt(x_7, self.W16(), self.quant(name="quant_x_7"), self.quant(name="quant_weights_16"), axes=1) + self.b16

        return a_16

class Q_Vgg16_Cifar10(Vgg16_Cifar10):

    def __init__(self, inp,m_init, k):
        """
        Instantiates a LeNet with quantized parameters and layer inputs. The quantizer does not
        clip to a fixed range.

        Args:
            inp: tf.tensor, the input of the network
            m_init: tf.initializer, the initializer for the resolution of the quantizers
            k: integer, the approximation order for approximate quantization
        """
        self.m_init = m_init
        self.k = k
        super(Q_Vgg16_Cifar10, self).__init__(inp)

    def linear(self, name):
        return UniformLinear(name=name)

    def quant(self, name):
        return ApproxUniformQuantizer(m_init=self.m_init, k=self.k, name=name)

    def _get_logits(self):
        # ------------------------create the network graph--------------------------
        # Conv layer 1:
        a_1 = qc(self.inp, self.W1(),self.quant(name="quant_input"), self.quant(name="quant_weights_1"), padding="SAME") + self.b1

        # Conv layer 2:
        a_2 = qc(a_1, self.W2(), self.quant(name="quant_a_1"), self.quant(name="quant_weights_2"), padding="SAME") + self.b2
        # max pooling 1: 16 x 16
        x_1 = tf.nn.max_pool(tf.nn.relu(a_2), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 3:
        a_3 = qc(x_1, self.W3(), self.quant(name="quant_x_1"), self.quant(name="quant_weights_3"), padding="SAME") + self.b3
        # Conv layer 4:
        a_4 = qc(a_3, self.W4(), self.quant(name="quant_a_3"), self.quant(name="quant_weights_4"), padding="SAME") + self.b4
        # max pooling 2: 8 x 8
        x_2 = tf.nn.max_pool(tf.nn.relu(a_4), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 5:
        a_5 = qc(x_2, self.W5(), self.quant(name="quant_x_2"), self.quant(name="quant_weights_5"), padding="SAME") + self.b5
        # Conv layer 6:
        a_6 = qc(a_5, self.W6(),self.quant(name="quant_a_5"), self.quant(name="quant_weights_6"), padding="SAME") + self.b6
        # Conv layer 7:
        a_7 = qc(a_6, self.W7(), self.quant(name="quant_a_6"), self.quant(name="quant_weights_7"), padding="SAME") + self.b7
        # max pooling 3: 4 x 4
        x_3 = tf.nn.max_pool(tf.nn.relu(a_7), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 8:
        a_8 = qc(x_3, self.W8(), self.quant(name="quant_x_3"), self.quant(name="quant_weights_8"), padding="SAME") + self.b8
        # Conv layer 9:
        a_9 = qc(a_8, self.W9(), self.quant(name="quant_a_9"), self.quant(name="quant_weights_9"), padding="SAME") + self.b9
        # Conv layer 10:
        a_10 = qc(a_9, self.W10(), self.quant(name="quant_a_10"), self.quant(name="quant_weights_10"), padding="SAME") + self.b10
        # max pooling 4: 2 x 2
        x_4 = tf.nn.max_pool(tf.nn.relu(a_10), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 11:
        a_11 = qc(x_4, self.W11(), self.quant(name="quant_x_4"), self.quant(name="quant_weights_11"), padding="SAME") + self.b11
        # Conv layer 12:
        a_12 = qc(a_11, self.W12(), self.quant(name="quant_a_11"), self.quant(name="quant_weights_12"), padding="SAME") + self.b12
        # Conv layer 13:
        a_13 = qc(a_12, self.W13(), self.quant(name="quant_a_12"), self.quant(name="quant_weights_13"), padding="SAME") + self.b13
        # max pooling 5: 1 x 1
        x_5 = tf.contrib.layers.flatten(tf.nn.max_pool(tf.nn.relu(a_13), [1, 2, 2, 1], [1, 2, 2, 1], "SAME"))

        # fc layer 14:
        a_14 = qt(x_5, self.W14(),self.quant(name="quant_x_5"), self.quant(name="quant_weights_14"), axes=1) + self.b14
        x_6 = tf.nn.relu(a_14)

        # output layer 15 returning the logits:
        a_15 = qt(x_6, self.W15(),self.quant(name="quant_x_6"), self.quant(name="quant_weights_15"), axes=1) + self.b15

        return a_15

class lin_Vgg16_Cifar10(Vgg16_Cifar10):

    def __init__(self, inp):
        """
        Instantiates a LeNet with quantized parameters and layer inputs. The quantizer does not
        clip to a fixed range.

        Args:
            inp: tf.tensor, the input of the network
            m_init: tf.initializer, the initializer for the resolution of the quantizers
            k: integer, the approximation order for approximate quantization
        """
        super(lin_Vgg16_Cifar10, self).__init__(inp)

    def linear(self, name):
        return UniformLinear(name=name)

    def quant(self, name):
        return UniformLinear(name=name)

    def _get_logits(self):
        # ------------------------create the network graph--------------------------
        # Conv layer 1:
        a_1 = qc(self.inp, self.W1(),self.quant(name="quant_input"), self.quant(name="quant_weights_1"), padding="SAME") + self.b1

        # Conv layer 2:
        a_2 = qc(a_1, self.W2(), self.quant(name="quant_a_1"), self.quant(name="quant_weights_2"), padding="SAME") + self.b2
        # max pooling 1: 16 x 16
        x_1 = tf.nn.max_pool(tf.nn.relu(a_2), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 3:
        a_3 = qc(x_1, self.W3(), self.quant(name="quant_x_1"), self.quant(name="quant_weights_3"), padding="SAME") + self.b3
        # Conv layer 4:
        a_4 = qc(a_3, self.W4(), self.quant(name="quant_a_3"), self.quant(name="quant_weights_4"), padding="SAME") + self.b4
        # max pooling 2: 8 x 8
        x_2 = tf.nn.max_pool(tf.nn.relu(a_4), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 5:
        a_5 = qc(x_2, self.W5(), self.quant(name="quant_x_2"), self.quant(name="quant_weights_5"), padding="SAME") + self.b5
        # Conv layer 6:
        a_6 = qc(a_5, self.W6(),self.quant(name="quant_a_5"), self.quant(name="quant_weights_6"), padding="SAME") + self.b6
        # Conv layer 7:
        a_7 = qc(a_6, self.W7(), self.quant(name="quant_a_6"), self.quant(name="quant_weights_7"), padding="SAME") + self.b7
        # max pooling 3: 4 x 4
        x_3 = tf.nn.max_pool(tf.nn.relu(a_7), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 8:
        a_8 = qc(x_3, self.W8(), self.quant(name="quant_x_3"), self.quant(name="quant_weights_8"), padding="SAME") + self.b8
        # Conv layer 9:
        a_9 = qc(a_8, self.W9(), self.quant(name="quant_a_8"), self.quant(name="quant_weights_9"), padding="SAME") + self.b9
        # Conv layer 10:
        a_10 = qc(a_9, self.W10(), self.quant(name="quant_a_9"), self.quant(name="quant_weights_10"), padding="SAME") + self.b10
        # max pooling 4: 2 x 2
        x_4 = tf.nn.max_pool(tf.nn.relu(a_10), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 11:
        a_11 = qc(x_4, self.W11(), self.quant(name="quant_x_4"), self.quant(name="quant_weights_11"), padding="SAME") + self.b11
        # Conv layer 12:
        a_12 = qc(a_11, self.W12(), self.quant(name="quant_a_11"), self.quant(name="quant_weights_12"), padding="SAME") + self.b12
        # Conv layer 13:
        a_13 = qc(a_12, self.W13(), self.quant(name="quant_a_12"), self.quant(name="quant_weights_13"), padding="SAME") + self.b13
        # max pooling 5: 1 x 1
        x_5 = tf.contrib.layers.flatten(tf.nn.max_pool(tf.nn.relu(a_13), [1, 2, 2, 1], [1, 2, 2, 1], "SAME"))

        # fc layer 14:
        a_14 = qt(x_5, self.W14(),self.quant(name="quant_x_5"), self.quant(name="quant_weights_14"), axes=1) + self.b14
        x_6 = tf.nn.relu(a_14)

        # fc layer 15 returning the logits:
        a_15 = qt(x_6, self.W15(),self.quant(name="quant_x_6"), self.quant(name="quant_weights_15"), axes=1) + self.b15
        x_7 = tf.nn.relu(a_15)

        # output layer 16 returning the logits:
        a_16 = qt(x_7, self.W16(), self.quant(name="quant_x_7"), self.quant(name="quant_weights_16"), axes=1) + self.b16

        return a_16

class QC_list_Vgg16_Cifar10(Vgg16_Cifar10):

    def __init__(self, inp, c_init_list, k, n_steps):
        """
        Instantiates a LeNet with quantized parameters and layer inputs. The quantizer does
        clip to a fixed range c.

        Args:
            inp: tf.tensor, the input of the network
            c_init: tf.initializer, the initializer for the range c of the quantizers with clipping
            k: integer), the approximation order for approximate quantization
            n_steps: positive odd integer (numpy), the number of quantization steps used for training
        """
        self.c_init = cycle(c_init_list)
        self.k = k
        self.n_steps = n_steps
        super(QC_list_Vgg16_Cifar10, self).__init__(inp)

    def linear(self, name):
        return UniformLinear(name=name)

    def quant(self, name, steps = None):
        if steps== None:
            return ClippedApproxUniformQuantizer(c_init=next(self.c_init), k=self.k, n_steps=self.n_steps, name=name)
        else:
            return ClippedApproxUniformQuantizer(c_init=next(self.c_init), k=self.k, n_steps=steps, name=name)

    def _get_logits(self):
        # ------------------------create the network graph--------------------------
        # Conv layer 1:
        a_1 = qc(self.inp, self.W1(),self.quant(name="quant_input",steps=128), self.quant(name="quant_weights_1"), padding="SAME") + self.b1

        # Conv layer 2:
        a_2 = qc(a_1, self.W2(), self.quant(name="quant_a_2"), self.quant(name="quant_weights_2"), padding="SAME") + self.b2
        # max pooling 1: 16 x 16
        x_1 = tf.nn.max_pool(tf.nn.relu(a_2), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 3:
        a_3 = qc(x_1, self.W3(), self.quant(name="quant_x_1"), self.quant(name="quant_weights_3"), padding="SAME") + self.b3
        # Conv layer 4:
        a_4 = qc(a_3, self.W4(), self.quant(name="quant_a_3"), self.quant(name="quant_weights_4"), padding="SAME") + self.b4
        # max pooling 2: 8 x 8
        x_2 = tf.nn.max_pool(tf.nn.relu(a_4), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 5:
        a_5 = qc(x_2, self.W5(), self.quant(name="quant_x_2"), self.quant(name="quant_weights_5"), padding="SAME") + self.b5
        # Conv layer 6:
        a_6 = qc(a_5, self.W6(),self.quant(name="quant_a_5"), self.quant(name="quant_weights_6"), padding="SAME") + self.b6
        # Conv layer 7:
        a_7 = qc(a_6, self.W7(), self.quant(name="quant_a_6"), self.quant(name="quant_weights_7"), padding="SAME") + self.b7
        # max pooling 3: 4 x 4
        x_3 = tf.nn.max_pool(tf.nn.relu(a_7), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 8:
        a_8 = qc(x_3, self.W8(), self.quant(name="quant_x_3"), self.quant(name="quant_weights_8"), padding="SAME") + self.b8
        # Conv layer 9:
        a_9 = qc(a_8, self.W9(), self.quant(name="quant_a_8"), self.quant(name="quant_weights_9"), padding="SAME") + self.b9
        # Conv layer 10:
        a_10 = qc(a_9, self.W10(), self.quant(name="quant_a_9"), self.quant(name="quant_weights_10"), padding="SAME") + self.b10
        # max pooling 4: 2 x 2
        x_4 = tf.nn.max_pool(tf.nn.relu(a_10), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 11:
        a_11 = qc(x_4, self.W11(), self.quant(name="quant_x_4"), self.quant(name="quant_weights_11"), padding="SAME") + self.b11
        # Conv layer 12:
        a_12 = qc(a_11, self.W12(), self.quant(name="quant_a_11"), self.quant(name="quant_weights_12"), padding="SAME") + self.b12
        # Conv layer 13:
        a_13 = qc(a_12, self.W13(), self.quant(name="quant_a_12"), self.quant(name="quant_weights_13"), padding="SAME") + self.b13
        # max pooling 5: 1 x 1
        x_5 = tf.contrib.layers.flatten(tf.nn.max_pool(tf.nn.relu(a_13), [1, 2, 2, 1], [1, 2, 2, 1], "SAME"))

        # fc layer 14:
        a_14 = qt(x_5, self.W14(),self.quant(name="quant_x_5"), self.quant(name="quant_weights_14"), axes=1) + self.b14
        x_6 = tf.nn.relu(a_14)

        # fc layer 15 returning the logits:
        a_15 = qt(x_6, self.W15(), self.quant(name="quant_x_6"), self.quant(name="quant_weights_15"), axes=1) + self.b15
        x_7 = tf.nn.relu(a_15)

        # output layer 16 returning the logits:
        a_16 = qt(x_7, self.W16(), self.quant(name="quant_x_7"), self.quant(name="quant_weights_16"), axes=1) + self.b16

        return a_16

class PartQC_list_Vgg16_Cifar10(Vgg16_Cifar10):

    def __init__(self, inp, c_init_list, k, n_steps):
        """
        Instantiates a LeNet with quantized parameters and layer inputs. The quantizer does
        clip to a fixed range c.

        Args:
            inp: tf.tensor, the input of the network
            c_init: tf.initializer, the initializer for the range c of the quantizers with clipping
            k: integer), the approximation order for approximate quantization
            n_steps: positive odd integer (numpy), the number of quantization steps used for training
        """
        self.c_init = cycle(c_init_list)
        self.k = k
        self.n_steps = n_steps
        super(PartQC_list_Vgg16_Cifar10, self).__init__(inp)

    def linear(self, name):
        return UniformLinear(name=name)

    def quant(self, name, steps = None):
        if steps== None:
            return ClippedApproxUniformQuantizer(c_init=next(self.c_init), k=self.k, n_steps=self.n_steps, name=name)
        else:
            return ClippedApproxUniformQuantizer(c_init=next(self.c_init), k=self.k, n_steps=steps, name=name)

    def _get_logits(self):
        # ------------------------create the network graph--------------------------
        # Conv layer 1:
        a_1 = qc(self.inp, self.W1(),self.quant(name="quant_input",steps= 256), self.quant(name="quant_weights_1", steps= 256), padding="SAME") + self.b1

        # Conv layer 2:
        a_2 = qc(a_1, self.W2(), self.linear(name="quant_a_2"), self.quant(name="quant_weights_2"), padding="SAME") + self.b2
        # max pooling 1: 16 x 16
        x_1 = tf.nn.max_pool(tf.nn.relu(a_2), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 3:
        a_3 = qc(x_1, self.W3(), self.linear(name="quant_x_1"), self.quant(name="quant_weights_3"), padding="SAME") + self.b3
        # Conv layer 4:
        a_4 = qc(a_3, self.W4(), self.linear(name="quant_a_3"), self.quant(name="quant_weights_4"), padding="SAME") + self.b4
        # max pooling 2: 8 x 8
        x_2 = tf.nn.max_pool(tf.nn.relu(a_4), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 5:
        a_5 = qc(x_2, self.W5(), self.linear(name="quant_x_2"), self.quant(name="quant_weights_5"), padding="SAME") + self.b5
        # Conv layer 6:
        a_6 = qc(a_5, self.W6(),self.linear(name="quant_a_5"), self.quant(name="quant_weights_6"), padding="SAME") + self.b6
        # Conv layer 7:
        a_7 = qc(a_6, self.W7(), self.linear(name="quant_a_6"), self.quant(name="quant_weights_7"), padding="SAME") + self.b7
        # max pooling 3: 4 x 4
        x_3 = tf.nn.max_pool(tf.nn.relu(a_7), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 8:
        a_8 = qc(x_3, self.W8(), self.linear(name="quant_x_3"), self.quant(name="quant_weights_8"), padding="SAME") + self.b8
        # Conv layer 9:
        a_9 = qc(a_8, self.W9(), self.linear(name="quant_a_8"), self.quant(name="quant_weights_9"), padding="SAME") + self.b9
        # Conv layer 10:
        a_10 = qc(a_9, self.W10(), self.linear(name="quant_a_9"), self.quant(name="quant_weights_10"), padding="SAME") + self.b10
        # max pooling 4: 2 x 2
        x_4 = tf.nn.max_pool(tf.nn.relu(a_10), [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

        # Conv layer 11:
        a_11 = qc(x_4, self.W11(), self.linear(name="quant_x_4"), self.quant(name="quant_weights_11"), padding="SAME") + self.b11
        # Conv layer 12:
        a_12 = qc(a_11, self.W12(), self.linear(name="quant_a_11"), self.quant(name="quant_weights_12"), padding="SAME") + self.b12
        # Conv layer 13:
        a_13 = qc(a_12, self.W13(), self.linear(name="quant_a_12"), self.quant(name="quant_weights_13"), padding="SAME") + self.b13
        # max pooling 5: 1 x 1
        x_5 = tf.contrib.layers.flatten(tf.nn.max_pool(tf.nn.relu(a_13), [1, 2, 2, 1], [1, 2, 2, 1], "SAME"))

        # fc layer 14:
        a_14 = qt(x_5, self.W14(),self.linear(name="quant_x_5"), self.quant(name="quant_weights_14"), axes=1) + self.b14
        x_6 = tf.nn.relu(a_14)

        # fc layer 15 returning the logits:
        a_15 = qt(x_6, self.W15(), self.linear(name="quant_x_6"), self.quant(name="quant_weights_15"), axes=1) + self.b15
        x_7 = tf.nn.relu(a_15)

        # output layer 16 returning the logits:
        a_16 = qt(x_7, self.W16(), self.linear(name="quant_x_7"), self.quant(name="quant_weights_16"), axes=1) + self.b16

        return a_16