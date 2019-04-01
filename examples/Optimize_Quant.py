"""
An example that learns the optimal approximate uniform symmetric mid-even quantizer for a given data distribution. 
We use Stochastic gradient descent for optimization of the range. The #steps used for quantization is a fixed design
parameter. We test it with:

    I)   Normal distributed data
    II)  Laplacian distributed data
    III) Uniform distributed data

Lukas Mauch
"""

import numpy as np
import tensorflow as tf
from bayesian_dnn.misc import setup
from bayesian_dnn.quantization import ClippedApproxUniformQuantizer as CAUQ
from scipy.stats import norm
from scipy.stats import laplace
from scipy.stats import uniform
import matplotlib.pyplot as plt

std  = 3.0
n_steps = 11

#--------------case I: Normal------------------
x    = tf.constant(std*np.random.randn(10000), dtype=tf.float32, name='x')
qac  = CAUQ(c_init=tf.constant_initializer(1.0, dtype=tf.float32), n_steps=n_steps, k=2, name='clipped_approx_quant')
xq   = qac(x)

#compute the mean squared error (mse) between quantized values xq and unquantized values x
mse_loss = tf.reduce_mean(tf.pow(x-xq, 2)) 

#set up the optimizer to optimize the range c of the quantizer for minimum mse
optimizer       = tf.train.GradientDescentOptimizer(2.0)
min_mse         = optimizer.minimize(mse_loss, var_list=qac.c)

#create the tf session, initialize all variables and optimize
sess = tf.Session(config=setup.config_tensorflow())
init = tf.global_variables_initializer()
sess.run(init)

for i in range(50):
    _, mse = sess.run((min_mse, mse_loss))
    print("MSE: " + str(mse))
   
c_opt = sess.run(qac.c)
print("optimal c: " + str(c_opt))
print(c_opt/std)
sess.close()
    
    
#plot the results
x   = np.linspace(-5*std,5*std,10000)
pdf = norm.pdf(x, scale=std)

plt.figure(1)
plt.plot(x, pdf)
plt.axvline(-c_opt, color="limegreen")
plt.axvline(c_opt, color="limegreen")
plt.xlabel("unquantized x")
plt.ylabel("p(x)")
plt.title("Normal distribution")
plt.show()

tf.reset_default_graph()


#--------------case II: Laplacian------------------
x    = tf.constant(np.random.laplace(scale=std, size=10000), dtype=tf.float32, name='x')
qac  = CAUQ(c_init=tf.constant_initializer(1.0, dtype=tf.float32), n_steps=n_steps, k=2, name='clipped_approx_quant')
xq   = qac(x)

#compute the mean squared error (mse) between quantized values xq and unquantized values x
mse_loss = tf.reduce_mean(tf.pow(x-xq, 2)) 

#set up the optimizer to optimize the range c of the quantizer for minimum mse
optimizer       = tf.train.GradientDescentOptimizer(0.2)
min_mse         = optimizer.minimize(mse_loss, var_list=qac.c)

#create the tf session, initialize all variables and optimize
sess = tf.Session(config=setup.config_tensorflow())
init = tf.global_variables_initializer()
sess.run(init)

for i in range(500):
    _, mse = sess.run((min_mse, mse_loss))
    print("MSE: " + str(mse))
   
c_opt = sess.run(qac.c)
print("optimal c: " + str(c_opt))
print(c_opt/std)
sess.close()
    
    
#plot the results
x   = np.linspace(-5*std,5*std,1000)
pdf = laplace.pdf(x, scale=std)

plt.figure(2)
plt.plot(x, pdf)
plt.axvline(-c_opt, color="limegreen")
plt.axvline(c_opt, color="limegreen")
plt.xlabel("unquantized x")
plt.ylabel("p(x)")
plt.title("Laplacian distribution")
plt.show()

tf.reset_default_graph()


#--------------case III: Uniform------------------
c    = std * np.sqrt(3)
x    = tf.constant(np.random.uniform(low=-c, high=c, size=1000), dtype=tf.float32, name='x')
qac  = CAUQ(c_init=tf.constant_initializer(1.0, dtype=tf.float32), n_steps=n_steps, k=2, alpha=0.5, name='clipped_approx_quant')
xq   = qac(x)

#compute the mean squared error (mse) between quantized values xq and unquantized values x
mse_loss = tf.reduce_mean(tf.pow(x-xq, 2)) 

#set up the optimizer to optimize the range c of the quantizer for minimum mse
optimizer       = tf.train.GradientDescentOptimizer(1.0)
min_mse         = optimizer.minimize(mse_loss, var_list=qac.c)

#create the tf session, initialize all variables and optimize
sess = tf.Session(config=setup.config_tensorflow())
init = tf.global_variables_initializer()
sess.run(init)

for i in range(50):
    _, mse = sess.run((min_mse, mse_loss))
    print("MSE: " + str(mse))
   
c_opt = sess.run(qac.c)
print("optimal c: " + str(c_opt))
print(c_opt/std)
sess.close()
    
    
#plot the results
x   = np.linspace(-5*std,5*std,1000)
pdf = uniform.pdf(x, loc=-c, scale=2*c)

plt.figure(3)
plt.plot(x, pdf)
plt.axvline(-c_opt, color="limegreen")
plt.axvline(c_opt, color="limegreen")
plt.xlabel("unquantized x")
plt.ylabel("p(x)")
plt.title("Uniform distribution")
plt.show()
