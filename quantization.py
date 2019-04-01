import tensorflow as tf
import math as mathlib


#-----------------------------some important methods for quantization------------------------------    
def clip_max(x, c, alpha):
    """
    The clipping function in the maximum saturation region 
    
    Args:
        c: tf.tensor in R, the range to clip to
        alpha: tf.tensor in R, the smoothing of the clipping
    """
    with tf.name_scope("clip_max"):
        return tf.abs(c) + alpha * tf.atan(x - tf.abs(c))

def clip_min(x, c, alpha):
    """
    The clipping function in the minimum saturation region
    
    Args:
        c: tf.tensor in R, the range to clip to
        alpha: tf.tensor in R, the smoothing of the clipping
    """
    with tf.name_scope("clip_min"):
        return -tf.abs(c) - alpha * tf.atan(tf.negative(x) - tf.abs(c))

def clip(x, c, alpha):
    """
    Uses soft clipping to clip x to the value range -c <= x <= c
    
    Args:
        c: tf.tensor in R, the range to clip to
        alpha: tf.tensor in R, the smoothing of the clipping
    """
    with tf.name_scope("create_masks"):
        max_clip_mask = tf.cast(x > tf.abs(c), dtype=tf.float32)
        min_clip_mask = tf.cast(x < tf.negative(tf.abs(c)), dtype=tf.float32)
        lin_clip_mask = tf.cast(x > tf.negative(tf.abs(c)), dtype=tf.float32) * tf.cast(x < tf.abs(c), dtype=tf.float32)

        return max_clip_mask * clip_max(x, c, alpha) + min_clip_mask * clip_min(x, c, alpha) + lin_clip_mask * x


#-----------------------------definition of quantization------------------------------    
class UniformLinear(object):
    def __init__(self, name):
        self.name=name
        
    def __call__(self, x):
        """
        Performs no quantization of the values in x, using the resolution m.
        
        Args:
            x: the tensor that should be quantized
        """
        x = tf.identity(x,name=self.name)
        tf.add_to_collection('QUANT_VAL_APPROX', x) # just to make sure we have this value
        return x
    
    
class UniformQuantizer(object):
     
    def __init__(self, m_init=None, name=None):
        """
        An exact uniform quantizer with resolution m.
        
        Args:
            m_init: tf.initializer, initializer for the resolution of the quantizer
        """
        if name is None:
            self.name = 'quant_'+str(len(tf.trainable_variables()))
        else:
            self.name = name
            
        if m_init is None:
            self.m = tf.get_variable('m_'+self.name, shape=(1,), initializer=tf.constant_initializer(1), dtype=tf.float32, trainable=False)
        else:
            self.m = tf.get_variable('m_'+self.name, shape=(1,), initializer=m_init, dtype=tf.float32, trainable=False)
            
        tf.add_to_collection('QUANT_RESOLUTION', self.m)
        
    def __call__(self, x):
        """
        Performs exact uniform quantization of the values in x, using the resolution m.
        
        Args:
            x: the tensor that should be quantized
        """
        with tf.name_scope("exact_uniform_quantization"):
            quantized = tf.identity(self.m * tf.floor(x / self.m + 0.5),name=self.name)
            # tf.add_to_collection('QUANT_VAL', quantized)
        
        return  quantized

    
    
class ApproxUniformQuantizer(UniformQuantizer):
     
     def __init__(self, m_init=None, k=1, name=None):
        """
        An approximate uniform quantizer with resolution m.
        
        Args:
            m_init: tf.initializer, initializer for the resolution of the quantizer
            k: uint, the approximation order
        """
        super(ApproxUniformQuantizer, self).__init__(m_init, name)
        self.k = int(k)
     
     def __call__(self, x):
        """
        Performs approximate uniform quantization of the values in x, using the resolution m.
        
        Args:
            x: the tensor that should be quantized
        """
        with tf.name_scope("fourier_series"):
            pi = tf.constant(mathlib.pi)
            quantized = x
            for i in range(1,self.k+1):
                quantized += (1.0/tf.constant(i, dtype=tf.float32)) * tf.sin(tf.constant(i, dtype=tf.float32)*pi*(2.0*x/self.m - 1)) / pi * self.m

            quantized = tf.identity(quantized,name=self.name)
            tf.add_to_collection('QUANT_VAL', quantized)

            return quantized
    

class ClippedApproxUniformQuantizer(ApproxUniformQuantizer):


    def __init__(self, c_init=None, n_steps=5, k=1, alpha=0.5, name=None):
        """
        An approximate uniform quantizer with values clipped to the range from -c <= x_q <= c. The resolution is
        adapted to get the number n_steps of quantization steps. Because we have a symmetric mid-even quantizer, n_steps has to be an odd
        integer.

        Args:
            c_init: tf.initializer, the initializer for the range c of the quantizer we clip the values to. The value of c has to be initialized positive!
            n_steps: odd numpy variable of type uint, the number of steps we use for quantization (n_steps=3,5,7,...)
            k: uint, the approximation order
        """
        self.k = int(k)

        self.n_steps = tf.constant(n_steps, dtype=tf.float32)
        if name is None:
            self.name = 'quant_'+str(len(tf.trainable_variables()))
        else:
            self.name = name

        if c_init is None:
            self.c = tf.get_variable('c_'+self.name, shape=(1,), initializer=tf.constant_initializer(0.1), dtype=tf.float32, trainable=False)
        else:
            self.c = tf.get_variable('c_'+self.name, shape=(1,), initializer=c_init, dtype=tf.float32, trainable=False)

        tf.add_to_collection('QUANT_RANGE', self.c)
        with tf.name_scope("update_m_"+self.name):
            self.m = tf.constant(2.0, dtype=tf.float32) * self.c / (self.n_steps - tf.constant(1.0, dtype=tf.float32))
        tf.add_to_collection('QUANT_RESOLUTION',self.m)

        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def __call__(self, x):
        """
        Performs approximate uniform quantization of the values in x, clipping them to the range -c <= x <= c, first. We calculate the resolution m to
        obtain a fixed number of self.n_steps quantization steps.

        Args:
            x: the tensor that should be quantized
        """
        #use soft clipping of the data x
        with tf.name_scope(self.name):
            clipped = clip(x, self.c, self.alpha)

            #call quantization from the super class
            quantized = super(ClippedApproxUniformQuantizer, self).__call__(clipped)
            tf.add_to_collection('QUANT_VAL_APPROX', quantized)

            return quantized


class ClippedUniformQuantizer(UniformQuantizer):

    def __init__(self,c_init=None,n_steps=5, alpha=0, name=None):
        """
        An approximate uniform quantizer with values clipped to the range from -c <= x_q <= c. The resolution is
        adapted to get the number n_steps of quantization steps. Because we have a symmetric mid-even quantizer, n_steps has to be an odd
        integer.

        Args:
            c_init: tf.initializer, the initializer for the range c of the quantizer we clip the values to. The value of c has to be initialized positive!
            k: uint, the approximation order
        """

        self.n_steps = tf.constant(n_steps, dtype=tf.float32)
        if name is None:
            self.name = 'quant_' + str(len(tf.trainable_variables()))
        else:
            self.name = name

        if c_init is None:
            self.c = tf.get_variable('c_' + self.name, shape=(1,), initializer=tf.constant_initializer(0.1),
                                     dtype=tf.float32, trainable=False)
        else:
            self.c = tf.get_variable('c_' + self.name, shape=(1,), initializer=c_init, dtype=tf.float32, trainable=False)

        tf.add_to_collection('QUANT_RANGE', self.c)
        with tf.name_scope("update_m"):
            self.m = tf.constant(2.0, dtype=tf.float32) * self.c / (self.n_steps - tf.constant(1.0, dtype=tf.float32))
            tf.add_to_collection('QUANT_RESOLUTION', self.m)
        # super(ClippedUniformQuantizer, self).__init__(m_init, name)
        self.alpha = tf.constant(alpha, dtype=tf.float32)

    def __call__(self, x):
        """
        Performs approximate uniform quantization of the values in x, clipping them to the range -c <= x <= c, first. We calculate the resolution m to
        obtain a fixed number of self.n_steps quantization steps.

        Args:
            x: the tensor that should be quantized
        """
        # use soft clipping of the data x
        clipped = clip(x, self.c, self.alpha)

        # call quantization from the super class
        quantized = super(ClippedUniformQuantizer, self).__call__(clipped)
        tf.add_to_collection('QUANT_VAL_UNIFORM', quantized)

        return quantized

#-----------------------------definition of quantized operations------------------------------
def quant_convolution(x, y, qx=UniformLinear, qy=UniformLinear, *args, **kwargs):
    """
    Convolution of x with y, using quantization qx and qy. 
    
    Args:
        x: tf.tensor, first argument of convolution
        y: tf.tensor, second argument of convolution
        qx: callable, Returns class instance of Uniform Quantizer. The callable takes name as an input.
        qy: callable, Returns class instance of Uniform Quantizer. The callable takes name as an input.
    """
    with tf.name_scope("quant_conv"):
        quant_x = qx
        quant_y = qy
        return tf.nn.convolution(quant_x(x), quant_y(y), *args, **kwargs)
    
    
def quant_tensordot(x, y, qx=UniformLinear, qy=UniformLinear, *args, **kwargs):
    """
    Tensordot of x with y, using quantization qx and qy. 
    
    Args:
        x: tf.tensor, first argument of convolution
        y: tf.tensor, second argument of convolution
        qx: callable, Returns class instance of Uniform Quantizer. The callable takes name as an input.
        qy: callable, Returns class instance of Uniform Quantizer. The callable takes name as an input.
    """
    with tf.name_scope("quant_tensDot"):
        quant_x = qx
        quant_y = qy
        return tf.tensordot(quant_x(x), quant_y(y), *args, **kwargs)


def quant_add(x, y, qx=UniformLinear, qy=UniformLinear, *args, **kwargs):
    """
    Tensor addition of x with y, using quantization qx and qy. 
    
    Args:
        x: tf.tensor, first argument of convolution
        y: tf.tensor, second argument of convolution
        qx: callable, Returns class instance of Uniform Quantizer. The callable takes name as an input.
        qy: callable, Returns class instance of Uniform Quantizer. The callable takes name as an input.
    """    
    quant_x = qx(name='quant_'+x.name[:-2])
    quant_y = qy(name='quant_'+y.name[:-2])
    return tf.add(quant_x(x), quant_y(y), *args, **kwargs)
    
    
    
#----------------------------------TEST----------------------------------
if __name__ == "__main__":

    from misc import setup
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    sess = tf.Session(config=setup.config_tensorflow())
    x    = tf.constant(np.linspace(-10.0, 10.0, 1000), dtype=tf.float32, name='x')
    y    = tf.constant(np.linspace(-10.0, 10.0, 1000), dtype=tf.float32, name='y')
    z    = tf.constant(np.linspace(-3, 3, 3), dtype=tf.float32, name='z')
    
    #exact uniform quantization
    qe   = UniformQuantizer(tf.constant_initializer(2), name='exact_quant')
    xqe  = qe(x)
    
    #approximate uniform quantization
    qa   = ApproxUniformQuantizer(tf.constant_initializer(2.0, dtype=tf.float32),k=5, name='approx_quant')
    xqa  = qa(x)
    
    #approximate uniform quantization with clipping
    qac  = ClippedApproxUniformQuantizer(c_init=tf.constant_initializer(5.0, dtype=tf.float32), n_steps=8, alpha=0,k=5, name='clipped_approx_quant')
    xqac = qac(x)
    
    #quantized sum of x and y
    sq   = quant_add(x, y, lambda name: UniformQuantizer(name=name), lambda name: UniformQuantizer(name=name))        
       
    #initialize and run
    init = tf.initialize_all_variables()
    sess.run(init)    
    
    res_e   = sess.run(xqe)
    res_a   = sess.run(xqa)
    res_ac  = sess.run(xqac)
    res_sq  = sess.run(sq)
    
    #print and plot the results     
    print(res_sq)

    print("The collection of resolutions: \n")
    print(tf.get_collection('QUANT_RESOLUTION'))

    plt.figure(1)
    # plt.plot(np.linspace(-10.0, 10.0,1000), res_e)
    # plt.plot(np.linspace(-10.0, 10.0,1000), res_a)
    plt.plot(np.linspace(-10, 10, 1000), res_ac,"g-")
    # plt.legend(['exact quantization', 'approximate quantization','clipped approximate quantization'])
    plt.legend(['alpha = 1'])
    plt.grid()
    plt.show()
    
    
    plt.figure(2)
    plt.plot(np.linspace(-10.0, 10.0,1000), res_ac)
    plt.legend(['clipped approximate quantization'])
    plt.grid()
    plt.show()
    
    #plot the gradients of the soft clipping function over clipping range and non-clipped values
    non_clipped = tf.placeholder(dtype=tf.float32)
    c           = tf.placeholder(dtype=tf.float32)
    alpha       = tf.constant(1.0)
    clipped     = clip(non_clipped, c, alpha)
    dclipped_dc = tf.gradients(clipped, c)
    
    n_bins = 200
    res = np.zeros((n_bins,n_bins))
    for i,nc_v in enumerate(np.linspace(-10,10,n_bins)):
        for j,c_v in enumerate(np.linspace(0,10,n_bins)):
            res[i,j] = sess.run(dclipped_dc, feed_dict={non_clipped: nc_v, c: c_v})[0]
    
    sess.close()
    
    fig = plt.figure(3)
    ax  = fig.gca(projection='3d')
    xv, yv = np.meshgrid(np.linspace(-10,10,n_bins), np.linspace(0,10,n_bins))
    surf = ax.plot_surface(xv,yv,res)
    plt.xlabel('non-clipped value')
    plt.ylabel('clipping range')
    plt.title('Soft Clipping')
    plt.show()
