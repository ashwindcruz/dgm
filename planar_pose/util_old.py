import pdb

import numpy as np
import math
import subprocess
from subprocess import PIPE

from chainer import cuda
from chainer.cuda import cupy
import chainer.functions as F

def print_compute_graph(file, g):
    format = file.split('.')[-1]
    cmd = 'dot -T%s -o %s'%(format,file)
    p=subprocess.Popen(cmd, stdin=PIPE, stdout=PIPE, stderr=PIPE, shell=True)
    p.stdin.write(g.dump())
    p.communicate()
    return p.returncode

def gaussian_kl_divergence_standard(mu, ln_var):
   """D_{KL}(N(mu,var) | N(0,1))"""
   batch_size = mu.data.shape[0]
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


    return logp_sum / batchsize

def gaussian_kl_divergence(z_0, z_0_mu, z_0_ln_var, z_T):
    """D_{KL}(q(z_0|x) || p(z_T))"""
    logp_q = gaussian_logp(z_0, z_0_mu, z_0_ln_var)
    logp_p = gaussian_logp(z_T, z_0_mu*0, z_0_ln_var/z_0_ln_var)
    kl_loss = logp_q - logp_p

    return kl_loss


