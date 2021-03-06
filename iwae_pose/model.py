import pdb

import math
import numpy as np
import time

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

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

    def decode(self, z):
        h = F.crelu(self.plin0(z))

        for i in range(self.num_layers-1):
            layer_name = 'plin' + str(i+1)
            h = F.crelu(self[layer_name](h))        

        self.pmu = self.plin_mu(h)
        self.pln_var = self.plin_ln_var(h)

        

    def __call__(self, x):
        # Obtain parameters for q(z|x)
        encoding_time = time.time()
        self.encode(x)
        encoding_time = float(time.time() - encoding_time)

        decoding_time_average = 0.

        xp = cuda.cupy
        self.importance_weights = 0
        self.w_holder = []
        self.kl = 0
        self.logp = 0

        for j in xrange(self.num_zsamples):
            # Sample z ~ q(z|x)
            z = F.gaussian(self.qmu, self.qln_var)

            # Compute log q(z|x)
            encoder_log = gaussian_logp(z, self.qmu, self.qln_var)
            
            # Obtain parameters for p(x|z)
            decoding_time = time.time()
            self.decode(z)
            decoding_time = time.time() - decoding_time
            decoding_time_average += decoding_time

            # Compute log p(x|z)
            decoder_log = gaussian_logp(x, self.pmu, self.pln_var)
            
            # Compute log p(z). The odd notation being used is to supply a mean of 0 and covariance of 1
            prior_log = gaussian_logp(z, self.qmu*0, self.qln_var/self.qln_var)
            
            # Store the latest log weight'
            current_temperature = min(self.temperature['value'],1.0)
            self.w_holder.append(decoder_log + current_temperature*(prior_log - encoder_log))

            # Store the KL and Logp equivalents. They are not used for computation but for recording and reporting. 
            self.kl += (encoder_log-prior_log)
            self.logp += (decoder_log)

        self.temperature['value'] += self.temperature['increment']
        

        # Compute w' for this sample (batch)
        logps = F.stack(self.w_holder)
        self.obj_batch = F.logsumexp(logps, axis=0) - np.log(self.num_zsamples)
        self.kl /= self.num_zsamples
        self.logp /= self.num_zsamples
        
        decoding_time_average /= self.num_zsamples
        
        batch_size = self.obj_batch.shape[0]
        
        self.obj = -F.sum(self.obj_batch)/batch_size        
        self.timing_info = np.array([encoding_time,decoding_time_average])

        return self.obj

