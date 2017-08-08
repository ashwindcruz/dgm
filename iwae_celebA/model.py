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
from util import gaussian_logp0
from util import bernoulli_logp

class VAE(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent, num_layers, temperature, num_zsamples=1):
       
        super(VAE, self).__init__()
        # initialise first encoder and decoder hidden layer separately because 
        # the input and output dims differ from the other hidden layers
        self.qlin0 = L.Linear(dim_in, dim_hidden)
        self.plin0 = L.Linear(dim_latent, dim_hidden)
        self._children.append('qlin0')
        self._children.append('plin0')

        # batch normalization layers at the start of the encoder and start of the decoder
        self.qlin_batch_norm_0 = L.BatchNormalization(dim_hidden)
        self.plin_batch_norm_0 = L.BatchNormalization(dim_hidden)
        self._children.append('qlin_batch_norm_0')
        self._children.append('plin_batch_norm_0')

        for i in range(num_layers-1):
            # encoder
            layer_name = 'qlin' + str(i+1)
            setattr(self, layer_name, L.Linear(2*dim_hidden, dim_hidden))
            self._children.append(layer_name) 

            layer_name = 'qlin_batch_norm_' + str(i+1)
            setattr(self, layer_name, L.BatchNormalization(dim_hidden))
            self._children.append(layer_name)

            # decoder
            layer_name = 'plin' + str(i+1)
            setattr(self, layer_name, L.Linear(2*dim_hidden, dim_hidden))
            self._children.append(layer_name)

            layer_name = 'plin_batch_norm_' + str(i+1)
            setattr(self, layer_name, L.BatchNormalization(dim_hidden))
            self._children.append(layer_name)

        # initialise the encoder and decoder output layer separately because
        # the input and output dims differ from the other hidden layers
        self.qlin_mu = L.Linear(2*dim_hidden, dim_latent)
        self.qlin_ln_var = L.Linear(2*dim_hidden, dim_latent)
        self.plin_ber_prob = L.Linear(2*dim_hidden, dim_in)
        self._children.append('qlin_mu')
        self._children.append('qlin_ln_var')
        self._children.append('plin_ber_prob')       

        self.num_layers = num_layers
        self.temperature = temperature
        self.num_zsamples = num_zsamples
        self.epochs_seen = 0

    def encode(self, x):
        h = self.qlin0(x)
        h = self.qlin_batch_norm_0(h)
        h = F.crelu(h)

        for i in range(self.num_layers-1):
            layer_name = 'qlin' + str(i+1)
            h = self[layer_name](h)

            layer_name = 'qlin_batch_norm_' + str(i+1)
            h = self[layer_name](h)
            h = F.crelu(h)
        
        self.qmu = self.qlin_mu(h)
        self.qln_var = self.qlin_ln_var(h)

    def decode(self, z):
        h = self.plin0(z)
        h = self.plin_batch_norm_0(h)
        h = F.crelu(h)

        for i in range(self.num_layers-1):
            layer_name = 'plin' + str(i+1)
            h = self[layer_name](h)

            layer_name = 'plin_batch_norm_' + str(i+1)
            h = self[layer_name](h)
            h = F.crelu(h)        

        self.p_ber_prob_logit = self.plin_ber_prob(h)

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
            decoder_log = bernoulli_logp(x, self.p_ber_prob_logit)
            
            # Compute log p(z). 
            prior_log = gaussian_logp0(z)
            
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