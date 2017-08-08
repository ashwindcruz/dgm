import pdb

import numpy as np
import math
import time

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from util import gaussian_kl_divergence_standard
from util import gaussian_logp0
from util import gaussian_logp

class VAE(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent, num_layers, num_trans, temperature, num_zsamples=1):
       
        super(VAE, self).__init__()
        # initialise first encoder and decoder hidden layer separately because 
        # the input and output dims differ from the other hidden layers
        self.qlin0 = L.Linear(dim_in, dim_hidden)
        self.plin0 = L.Linear(dim_latent, dim_hidden)
        self.qlin_h = L.Linear(2*dim_hidden, dim_latent)
        self._children.append('qlin0')
        self._children.append('plin0')
        self._children.append('qlin_h')

        for i in range(num_layers-1):
            # encoder
            layer_name = 'qlin' + str(i+1)
            setattr(self, layer_name, L.Linear(2*dim_hidden, dim_hidden))
            self._children.append(layer_name) 

            # decoder
            layer_name = 'plin' + str(i+1)
            setattr(self, layer_name, L.Linear(2*dim_hidden, dim_hidden))
            self._children.append(layer_name)

        # initialise the encoder and decoder output layer separately because
        # the input and output dims differ from the other hidden layers
        self.qlin_mu = L.Linear(2*dim_hidden, dim_latent)
        self.qlin_ln_var = L.Linear(2*dim_hidden, dim_latent)
        self.plin_ln_var = L.Linear(2*dim_hidden, dim_in)
        self.plin_mu = L.Linear(2*dim_hidden, dim_in)
        self._children.append('qlin_mu')
        self._children.append('qlin_ln_var')
        self._children.append('plin_mu')
        self._children.append('plin_ln_var')       
        
        # iaf transformations
        for i in range(num_trans):
            layer_name = 'qiaf_a' + str(i+1)
            setattr(self, layer_name, L.Linear(2*dim_latent, dim_latent))
            self._children.append(layer_name) 

            layer_name = 'qiaf_b' + str(i+1)
            setattr(self, layer_name, L.Linear(2*dim_latent, 2*dim_latent))
            self._children.append(layer_name)

        self.num_layers = num_layers
        self.num_trans = num_trans
        self.temperature = temperature
        self.num_zsamples = num_zsamples
        self.epochs_seen = 0

    def encode(self, x):
        h = F.crelu(self.qlin0(x))

        for i in range(self.num_layers-1):
            layer_name = 'qlin' + str(i+1)
            h = F.crelu(self[layer_name](h))
        
        self.qmu = self.qlin_mu(h)
        self.qln_var = self.qlin_ln_var(h)
        self.qh = self.qlin_h(h)

    def decode(self, z):
        h = F.crelu(self.plin0(z))

        for i in range(self.num_layers-1):
            layer_name = 'plin' + str(i+1)
            h = F.crelu(self[layer_name](h))        

        self.pmu = self.plin_mu(h)
        self.pln_var = self.plin_ln_var(h)

    def iaf(self, z, h, lin1, lin2):
        ms = F.crelu(lin1(F.concat((z, h), axis=1)))
        ms = lin2(ms)
        m, s = F.split_axis(ms, 2, axis=1)
        s = F.sigmoid(s)
        z = s*z + (1-s)*m
        # pdb.set_trace()
        return z, -F.sum(F.log(s), axis=1)

    def __call__(self, x):
        # Obtain parameters for q(z|x)
        encoding_time = time.time()
        self.encode(x)
        encoding_time = float(time.time() - encoding_time)

        self.logp_xz = 0
        self.logq = 0

        # For reporting purposes only
        self.logp = 0
        self.kl = 0

        decoding_time_average = 0.

        current_temperature = min(self.temperature['value'],1.0)
        self.temperature['value'] += self.temperature['increment']
        
        for j in xrange(self.num_zsamples):
            # z ~ q(z|x)
            z = F.gaussian(self.qmu, self.qln_var)

            decoding_time = time.time()

            # Apply inverse autoregressive flow (IAF)
            self.logq += gaussian_logp(z, self.qmu, self.qln_var)    # - log q(z|x)

            for i in range(self.num_trans):
                a_layer_name = 'qiaf_a' + str(i+1)
                b_layer_name = 'qiaf_b' + str(i+1)
                z, delta_logq = self.iaf(z, self.qh, self[a_layer_name], self[b_layer_name])
                self.logq += delta_logq

            self.logq *= current_temperature

            # Compute p(x|z)
            self.decode(z)

            decoding_time = time.time() - decoding_time
            decoding_time_average += decoding_time

            # Compute objective, p(x,z)
            logx_given_z = gaussian_logp(x, self.pmu, self.pln_var) # p(x|z)
            logz = (current_temperature*gaussian_logp0(z)) # p(z)
            self.logp_xz += (logx_given_z + logz)

            # For reporting purposes only
            self.logp += logx_given_z
            self.kl += (self.logq - logz) 

        decoding_time_average /= self.num_zsamples
        # self.logp_xz /= self.num_zsamples
        # self.logq /= self.num_zsamples

        # For reporting purposes only
        self.logp /= self.num_zsamples
        self.kl /= self.num_zsamples
        
        self.obj_batch = self.logp_xz - self.logq     # variational free energy
        self.obj_batch /= self.num_zsamples
        self.timing_info = np.array([encoding_time,decoding_time_average])

        batch_size = self.obj_batch.shape[0]
        self.obj = -F.sum(self.obj_batch)/batch_size  

        return self.obj

