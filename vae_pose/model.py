import pdb

import numpy as np
import math
import time

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from util import gaussian_kl_divergence_standard
from util import gaussian_logp


class VAE(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent, num_layers, temperature, num_zsamples=1):
       
        super(VAE, self).__init__()
        # initialise first encoder and decoder hidden layer separately because 
        # the input and output dims differ from the other hidden layers
        self.qlin0 = L.Linear(dim_in, dim_hidden)
        self.plin0 = L.Linear(dim_latent, dim_hidden)
        self._children.append('qlin0')
        self._children.append('plin0')

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

        self.num_layers = num_layers
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

        return self.qmu, self.qln_var

    def decode(self, z):
        h = F.crelu(self.plin0(z))

        for i in range(self.num_layers-1):
            layer_name = 'plin' + str(i+1)
            h = F.crelu(self[layer_name](h))        

        self.pmu = self.plin_mu(h)
        self.pln_var = self.plin_ln_var(h)

        return self.pmu, self.qln_var

    def __call__(self, x):
        # Compute q(z|x)
        encoding_time = time.time()
        self.encode(x)
        encoding_time = float(time.time() - encoding_time)

        decoding_time_average = 0.

        self.kl = gaussian_kl_divergence_standard(self.qmu, self.qln_var)
        self.logp = 0
        for j in xrange(self.num_zsamples):
            # z ~ q(z|x)
            z = F.gaussian(self.qmu, self.qln_var)

            # Compute p(x|z)
            decoding_time = time.time()
            self.decode(z)
            decoding_time = time.time() - decoding_time
            decoding_time_average += decoding_time

            # Compute objective
            self.logp += gaussian_logp(x, self.pmu, self.pln_var)

        current_temperature = min(self.temperature['value'],1.0)
        self.temperature['value'] += self.temperature['increment']
        # pdb.set_trace()
        decoding_time_average /= self.num_zsamples
        self.logp /= self.num_zsamples
        self.obj_batch = self.logp - (current_temperature*self.kl)
        self.timing_info = np.array([encoding_time,decoding_time_average])

        batch_size = self.obj_batch.shape[0]
        
        self.obj = -F.sum(self.obj_batch)/batch_size
        
        return self.obj

