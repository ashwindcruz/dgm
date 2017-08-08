import pdb

import numpy as np
import math
import time

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

from util import gaussian_kl_divergence
from util import gaussian_logp
from util import gaussian_logp0
from util import bernoulli_logp

class VAE(chainer.Chain):
    def __init__(self, dim_in, dim_hidden, dim_latent, num_layers, num_trans, temperature, num_zsamples=1):
       
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
        self.plin_ber_prob = L.Linear(2*dim_hidden, dim_in)
        self._children.append('qlin_mu')
        self._children.append('qlin_ln_var')
        self._children.append('plin_ber_prob')       

        # v0 and linear layer required for v_t of Householder flow transformations
        self.qlin_h_vec_0 = L.Linear(2*dim_hidden, dim_latent)
        self.qlin_h_vec_t = L.Linear(dim_latent, dim_latent)
        self._children.append('qlin_h_vec_0')
        self._children.append('qlin_h_vec_t')

        self.num_layers = num_layers
        self.num_trans = num_trans
        self.temperature = temperature
        self.num_zsamples = num_zsamples
        self.epochs_seen = 0
        # pdb.set_trace()

    def encode(self, x):
        h = F.crelu(self.qlin0(x))

        for i in range(self.num_layers-1):
            layer_name = 'qlin' + str(i+1)
            h = F.crelu(self[layer_name](h))
        
        self.qmu = self.qlin_mu(h)
        self.qln_var = self.qlin_ln_var(h)
        self.qh_vec_0 = self.qlin_h_vec_0(h)

        return self.qmu, self.qln_var, self.qh_vec_0

    def decode(self, z):
        h = F.crelu(self.plin0(z))

        for i in range(self.num_layers-1):
            layer_name = 'plin' + str(i+1)
            h = F.crelu(self[layer_name](h))        

        self.p_ber_prob_logit = self.plin_ber_prob(h)

        return self.p_ber_prob_logit
    
    def house_transform(self,z):
        vec_t = self.qh_vec_0
        
        for i in range(self.num_trans):
            vec_t = F.identity(self.qlin_h_vec_t(vec_t))
            vec_t_product = F.matmul(vec_t, vec_t, transb=True)
            vec_t_norm_sqr = F.tile(F.sum(F.square(vec_t)), (z.shape[0], z.shape[1]))
            z = z - 2*F.matmul(vec_t_product,  z)/vec_t_norm_sqr
        return z

    def __call__(self, x):
        # Obtain parameters for q(z|x)
        encoding_time = time.time()
        qmu, qln_var, qh_vec_0 = self.encode(x)
        encoding_time = float(time.time() - encoding_time)

        decoding_time_average = 0.

        self.kl = 0
        self.logp = 0
        for j in xrange(self.num_zsamples):
            # z_0 ~ q(z|x)
            z_0 = F.gaussian(qmu, qln_var)

            # Perform Householder flow transformation, Equation (8)
            decoding_time = time.time()
            z_T = self.house_transform(z_0)

            # Obtain parameters for p(x|z_T)
            p_ber_prob_logit = self.decode(z_T)
            decoding_time = time.time() - decoding_time
            decoding_time_average += decoding_time

            # Compute objective
            self.logp += bernoulli_logp(x, self.p_ber_prob_logit)
            self.kl += gaussian_kl_divergence(z_0, qmu, qln_var, z_T)
            
        
        decoding_time_average /= self.num_zsamples
        
        self.logp /= self.num_zsamples
        self.kl /= self.num_zsamples
        
        current_temperature = min(self.temperature['value'],1.0)
        self.obj_batch = self.logp - (current_temperature*self.kl)
        self.temperature['value'] += self.temperature['increment']


        self.timing_info = np.array([encoding_time,decoding_time_average])

        batch_size = self.obj_batch.shape[0]
        
        self.obj = -F.sum(self.obj_batch)/batch_size
        
        return self.obj

