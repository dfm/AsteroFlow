
# coding: utf-8

# In[1]:

# get_ipython().magic('matplotlib inline')
# get_ipython().magic('config IPython.matplotlib.backend = "retina"')
# from matplotlib import rcParams
# rcParams["figure.dpi"] = 150
# rcParams["savefig.dpi"] = 150


# In[2]:

import numpy as np
import matplotlib.pyplot as plt

from astropy.stats import LombScargle


# In[3]:

import tensorflow as tf

session = tf.get_default_session()
if session is None:
    session = tf.InteractiveSession()


# In[4]:

# These functions are used to transform bounded parameters to parameters to parameters with infinite range
def get_param_for_value(value, min_value, max_value):
    if np.any(value <= min_value) or np.any(value >= max_value):
        raise ValueError("value must be in the range (min_value, max_value)")
    return np.log(value - min_value) - np.log(max_value - value)

def get_value_for_param(param, min_value, max_value):
    return min_value + (max_value - min_value) / (1.0 + np.exp(-param))

def get_bounded_variable(name, value, min_value, max_value, dtype=tf.float64):
    param = tf.Variable(get_param_for_value(value, min_value, max_value), dtype=dtype, name=name + "_param")
    var = min_value + (max_value - min_value) / (1.0 + tf.exp(-param))
    log_jacobian = tf.log(var - min_value) + tf.log(max_value - var) - np.log(max_value - min_value)
    return param, var, tf.reduce_sum(log_jacobian), (min_value, max_value)

# This function constrains a pair of parameters to be a unit vector
def get_unit_vector(name, x_value, y_value, dtype=tf.float64):
    x_param = tf.Variable(x_value, dtype=dtype, name=name + "_x_param")
    y_param = tf.Variable(y_value, dtype=dtype, name=name + "_y_param")
    norm = tf.square(x_param) + tf.square(y_param)
    log_jacobian = -0.5*tf.reduce_sum(norm)
    norm = tf.sqrt(norm)
    x = x_param / norm
    y = y_param / norm
    return x_param, y_param, x, y, log_jacobian


# In[25]:

np.random.seed(42)
t = np.linspace(0, 6, 5000)[:3000]
yerr = 0.1 + np.zeros_like(t)
ivar = 1.0 / yerr**2

T = tf.float64

log_prior = tf.constant(0.0, dtype=T)
nl = 2
nn = 3
nmodes = (2*nn+1)*nl

# The peak power is a nu_max
log_numax_param, log_numax, log_jac, log_numax_range = get_bounded_variable("log_numax", np.log(200.0), np.log(150.0), np.log(300.0), dtype=T)
log_prior += log_jac

# The highest frequency for mode l is at nu_max_l
log_numax_l_param, log_numax_l, log_jac, log_numax_l_range = get_bounded_variable("log_numax_l", np.log(200.0) + np.zeros(nl), np.log(150.0), np.log(300.0), dtype=T)
log_prior += log_jac

# Frequency spacing
log_dnu_param, log_dnu, log_jac, log_dnu_range = get_bounded_variable("log_dnu", np.log(17.0), np.log(15.0), np.log(30.0), dtype=T)
log_prior += log_jac

# The phase for each mode
cp, sp = np.random.randn(2, nmodes)
phi_x, phi_y, cosphi, sinphi, log_jac = get_unit_vector("phi", cp, sp, dtype=T)
log_prior += log_jac
phi = tf.atan2(sinphi, cosphi)

# The parameters of the envelope
log_amp_param, log_amp, log_jac, log_amp_range = get_bounded_variable("log_amp", np.log(np.random.uniform(0.015, 0.02, nl)), np.log(0.01), np.log(0.03), dtype=T)
log_prior += log_jac
log_width = tf.Variable(np.log(25.0), dtype=T, name="log_width")
curve = tf.Variable(0.001, dtype=T, name="curve")

# Initialize
session.run(tf.global_variables_initializer())

# Define the frequency comb
dn = np.arange(-nn, nn+1)
l = np.arange(nl)
numax = tf.exp(log_numax)
numax_l = tf.exp(log_numax_l)
dnu = tf.exp(log_dnu)
nu = (numax_l + 0.5*l*dnu)[:, None] + dnu*(dn + curve*dn**2)[None, :]
amp = tf.exp(log_amp[:, None]-0.5*(nu - numax)**2 * tf.exp(-2*log_width))

# The echelle diagram
echelle = [tf.transpose(tf.mod(nu, dnu)), tf.transpose(nu)]

# Define the model
amp_flat = tf.reshape(amp, (-1,))
nu_flat = tf.reshape(nu, (-1,))
phi_flat = tf.reshape(phi, (-1,))
models = amp_flat[None, :] * tf.sin(2*np.pi*nu_flat[None, :]*t[:, None] + phi_flat[None, :])

model_y = tf.reduce_sum(models, axis=1)

# Simulate data from the real model
y = session.run(model_y) + yerr * np.random.randn(len(t))

log_like = -0.5*tf.reduce_sum(tf.square((y - model_y)/yerr))

var_list = [log_numax_param, log_numax_l_param, log_dnu_param, phi_x, phi_y, log_amp_param, log_width, curve]

log_prob = log_prior + log_like
grad = tf.gradients(log_prob, var_list)
session.run([log_prob, grad])

# In[26]:

import time
K = 50
strt = time.time()
for k in range(K):
    session.run(log_prob)
print((time.time() - strt) / K)

strt = time.time()
for k in range(K):
    session.run(grad)
print((time.time() - strt) / K)
assert 0


# In[7]:

neg_log_prob = -log_prob

opt = tf.contrib.opt.ScipyOptimizerInterface(neg_log_prob, var_list=[phi_x, phi_y])
opt.minimize(session)

opt = tf.contrib.opt.ScipyOptimizerInterface(neg_log_prob, var_list=[log_numax_param])
opt.minimize(session)

opt = tf.contrib.opt.ScipyOptimizerInterface(neg_log_prob, var_list=[log_numax_l_param])
opt.minimize(session)

opt = tf.contrib.opt.ScipyOptimizerInterface(neg_log_prob, var_list=[log_dnu_param])
opt.minimize(session)

opt = tf.contrib.opt.ScipyOptimizerInterface(neg_log_prob, var_list=[log_amp_param])
opt.minimize(session)

opt = tf.contrib.opt.ScipyOptimizerInterface(neg_log_prob, var_list=[log_width])
opt.minimize(session)

opt = tf.contrib.opt.ScipyOptimizerInterface(neg_log_prob, var_list=[curve])
opt.minimize(session)

opt = tf.contrib.opt.ScipyOptimizerInterface(neg_log_prob, var_list=var_list)
opt.minimize(session)


# In[8]:

m = LombScargle(t, y, yerr)
f, p = m.autopower(minimum_frequency=100, maximum_frequency=300, nyquist_factor=100, normalization="psd")
plt.plot(f, p);

m = LombScargle(t, session.run(model_y), yerr)
f, p = m.autopower(minimum_frequency=100, maximum_frequency=300, nyquist_factor=100, normalization="psd")
plt.plot(f, p);


# In[9]:

from helpers import TFModel
model = TFModel(log_prob, var_list)
model.setup(session)


# In[10]:

# We'll use the inverse Hessian to estimate the initial scales of the problem
hess = session.run(tf.hessians(log_prob, var_list))
var = 1.0 / np.abs(np.concatenate([np.diag(np.atleast_2d(h)) for h in hess]))


# In[11]:

import hemcee

metric = hemcee.metric.DenseMetric(np.diag(var))

sampler = hemcee.NoUTurnSampler(model.value, model.gradient, metric=metric)

q, lp = sampler.run_warmup(model.current_vector(), 10000)


# In[12]:

metric.sample_p()


# In[13]:

nuts = sampler.run_mcmc(q, 10000)


# In[14]:

plt.plot(nuts[0][:, 0]);


# In[18]:

import corner
corner.corner(nuts[0][:, -4:]);


# In[19]:

chain = nuts[0]


# In[20]:

from emcee.autocorr import integrated_time
tau_nuts = integrated_time(chain[:, None, :])
neff_nuts = len(chain) / np.mean(tau_nuts)
tau_nuts, neff_nuts


# In[24]:

nu_max_value = np.exp(get_value_for_param(nuts[0][:, 0], *log_numax_range))
dnu_value = np.exp(get_value_for_param(nuts[0][:, 3], *log_dnu_range))

corner.corner(np.vstack((nu_max_value, dnu_value)).T);


# In[ ]:



