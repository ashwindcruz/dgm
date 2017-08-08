import pdb

import numpy as np
import math
import subprocess
from subprocess import PIPE
import time

import chainer
from chainer import cuda
from chainer.cuda import cupy
import chainer.functions as F
from chainer.functions.activation import softplus
#import cuda

xp = cuda.cupy

def print_compute_graph(file, g):
    format = file.split('.')[-1]
    cmd = 'dot -T%s -o %s'%(format,file)
    p=subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    p.stdin.write(g.dump())
    p.communicate()
    return p.returncode

def gaussian_kl_divergence_standard(mu, ln_var):
   """D_{KL}(N(mu,var) | N(0,1))"""
   batch_size = float(mu.data.shape[0])
   S = F.exp(ln_var)
   D = mu.data.size
   
   KL_sum = 0.5*(F.sum(S, axis=1) + F.sum(mu*mu, axis=1) - F.sum(ln_var, axis=1) - D/batch_size)

   return KL_sum #/ batchsize

def gaussian_logp(x, mu, ln_var):
    """log N(x ; mu, var)"""
    batch_size = mu.data.shape[0]
    D = x.data.size
    S = F.exp(ln_var)
    xc = x - mu

    logp_sum = -0.5*(F.sum((xc*xc) / S, axis=1) + F.sum(ln_var, axis=1)
        + D/batch_size*math.log(2.0*math.pi))

    return logp_sum

def gaussian_logp0(x):
    """log N(x ; 0, 1)"""
    D = x.data.size
    batch_size = x.data.shape[0]

    logp_sum = -0.5*(F.sum(x*x, axis=1) + D/batch_size*math.log(2.0*math.pi))
    return logp_sum 

def gaussian_kl_divergence(z_0, z_0_mu, z_0_ln_var, z_T):
    """D_{KL}(q(z_0|x) || p(z_T))"""
    logp_q = gaussian_logp(z_0, z_0_mu, z_0_ln_var)
    logp_p = gaussian_logp(z_T, z_0_mu*0, z_0_ln_var/z_0_ln_var)
    kl_loss = logp_q - logp_p

    return kl_loss

def bernoulli_logp(x, ber_prob_logit):
    """logB(x;p)"""

    logp = softplus.softplus(ber_prob_logit) - x * ber_prob_logit
    logp = F.sum(logp, axis=1) 
    # pdb.set_trace()
    return -logp

# Function to evaluate average ELBO, SEM and timing for a particular dataset. 
def evaluate_dataset(vae_model, dataset, batch_size, log_file, backward, opt):
  volatile = 'ON'
  if(backward):
    volatile = 'OFF'

  N = dataset.shape[0]
  elbo = xp.zeros([N])
  kl = xp.zeros([N])
  logp = xp.zeros([N])
  backward_timing = np.array([0.])
  timing_info = np.array([0.,0.])

  for i in range(0,N/batch_size):
    data_subset = chainer.Variable(xp.asarray(dataset[i*batch_size:(i+1)*batch_size,:], dtype=np.float32), volatile=volatile)
    obj = vae_model(data_subset)

    # Exit early if there was a NaN in the batch, pops up for the IWAE
    if(math.isnan(obj.data)):
      return vae_model

    elbo[i*batch_size:(i+1)*batch_size] = vae_model.obj_batch.data
    kl[i*batch_size:(i+1)*batch_size] = vae_model.kl.data
    logp[i*batch_size:(i+1)*batch_size] = vae_model.logp.data
    
    timing_info += vae_model.timing_info

    if(backward):
      vae_model.zerograds()
      backward_timing_now = time.time()
      obj.backward()
      opt.update()
      backward_timing += (time.time() - backward_timing_now)


  # One final smaller batch to cover what couldn't be captured in the loop
  data_subset = chainer.Variable(xp.asarray(dataset[(N/batch_size)*batch_size:,:], dtype=np.float32), volatile=volatile)
  obj = vae_model(data_subset)

  # Exit early if there was a NaN in the batch, pops up for the IWAE
  if(math.isnan(obj.data)):
    return vae_model

  elbo[(N/batch_size)*batch_size:] = vae_model.obj_batch.data
  kl[(N/batch_size)*batch_size:] = vae_model.kl.data
  logp[(N/batch_size)*batch_size:] = vae_model.logp.data

  if(backward):
      vae_model.zerograds()
      #backward_timing_now = time.time()
      vae_model.obj.backward()
      opt.update()
      #backward_timing += (time.time() - backward_timing_now) # commented out since it shouldn't factor into the timing calculations

  # Don't use the latest timing information as it is a different batch size. Think more about this. 
  # TODO
  timing_info /= (N/batch_size)
  backward_timing /= (N/batch_size)

  # Calculate the average ELBO and the SEM
  obj_ave = elbo.mean()
  kl_mean  = kl.mean()
  logp_mean = logp.mean()
  obj_std = elbo.std()
  obj_sem = obj_std/xp.sqrt(N)

  with open(log_file, 'a') as f:
    f.write(str(obj_ave) + ',' + str(kl_mean) + ',' + str(logp_mean) + ',' + str(obj_sem) + ',' \
      + str(timing_info[0]) + ',' + str(timing_info[1]) + ',' + str(backward_timing[0]) + '\n')

  return vae_model