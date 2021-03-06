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

        # auxiliary variable
        self.qlina0 = L.Linear(dim_in, dim_hidden)
        self.plina0 = L.Linear(dim_latent, dim_hidden)
        self._children.append('qlina0')
        self._children.append('plina0')

        # z and x variable
        self.qlinz0 = L.Linear(dim_in+dim_latent, dim_hidden)
        self.plinx0 = L.Linear(dim_latent+dim_latent, dim_hidden)
        self._children.append('qlinz0')
        self._children.append('plinx0')


        # Set up the auxiliary inference model q(a|x) and the latent inference model q(z|a,x)
        for i in range(num_layers-1):
            # encoder for a
            layer_name = 'qlina' + str(i+1)
            setattr(self, layer_name, L.Linear(2*dim_hidden, dim_hidden))
            self._children.append(layer_name) 

            # decoder for a
            layer_name = 'plina' + str(i+1)
            setattr(self, layer_name, L.Linear(2*dim_hidden, dim_hidden))
            self._children.append(layer_name)

            # encoder for z
            layer_name = 'qlinz' + str(i+1)
            setattr(self, layer_name, L.Linear(2*dim_hidden, dim_hidden))
            self._children.append(layer_name)

            # decoder for z
            layer_name = 'plinx' + str(i+1)
            setattr(self, layer_name, L.Linear(2*dim_hidden, dim_hidden))
            self._children.append(layer_name)

        # initialise the encoder and decoder output layer separately because
        # the input and output dims differ from the other hidden layers
        self.qlina_mu = L.Linear(2*dim_hidden, dim_latent)
        self.qlina_ln_var = L.Linear(2*dim_hidden, dim_latent)
        self.qlinz_mu = L.Linear(2*dim_hidden, dim_latent)
        self.qlinz_ln_var = L.Linear(2*dim_hidden, dim_latent)
        self.plina_mu = L.Linear(2*dim_hidden, dim_latent)
        self.plina_ln_var = L.Linear(2*dim_hidden, dim_latent)
        self.plinx_ber_prob = L.Linear(2*dim_hidden, dim_in)
        
        self._children.append('qlina_mu')
        self._children.append('qlina_ln_var')
        self._children.append('qlinz_mu')
        self._children.append('qlinz_ln_var')
        self._children.append('plina_mu')
        self._children.append('plina_ln_var')
        self._children.append('plinx_ber_prob')

        self.num_layers = num_layers
        self.temperature = temperature
        self.num_zsamples = num_zsamples
        self.epochs_seen = 0

    def encode_a(self, x):
        a_params = F.crelu(self.qlina0(x))

        for i in range(self.num_layers-1):
            layer_name = 'qlina' + str(i+1)
            a_params = F.crelu(self[layer_name](a_params))

        self.qmu_a = self.qlina_mu(a_params)
        self.qln_var_a = self.qlina_ln_var(a_params)

        return self.qmu_a, self.qln_var_a

    def encode_z(self, x, a):
        # a = F.gaussian(self.qmu_a, self.qln_var_a) # This should be outside the encoding function. Pass the function a. 
        net_input = F.concat((x,a), axis=1)

        h = F.crelu(self.qlinz0(net_input))
        for i in range(self.num_layers-1):
            layer_name = 'qlinz' + str(i+1)
            h = F.crelu(self[layer_name](h))

        self.qmu_z = self.qlinz_mu(h)
        self.qln_var_z = self.qlinz_ln_var(h)

        return self.qmu_z, self.qln_var_z

    def decode_a(self, z):
        # net_input = F.concat((x,z), axis=1)
        
        h = F.crelu(self.plina0(z))

        for i in range(self.num_layers-1):
            layer_name = 'plina' + str(i+1)
            h = F.crelu(self[layer_name](h))        

        self.pmu_a = self.plina_mu(h)
        self.pln_var_a = self.plina_ln_var(h)

        return self.pmu_a, self.pln_var_a

    def decode(self,z):
        # pdb.set_trace()
        a = self.a_enc

        # If this function is coming from the sampling call, the batch size of z and a won't match. Manually handle that here.
        if (a.shape[0]!=z.shape[0]):
            a.volatile = 'ON'
            batch_size = z.shape[0]
            a.data = a.data[0:batch_size,:]

        net_input = F.concat((z,a), axis=1)

        h = F.crelu(self.plinx0(net_input))

        for i in range(self.num_layers-1):
            layer_name = 'plinx' + str(i+1)
            h = F.crelu(self[layer_name](h))

        self.p_ber_prob_logit = self.plinx_ber_prob(h)
        
        return self.p_ber_prob_logit

    def __call__(self, x):
        # Compute parameters for q(z|x, a)
        encoding_time_1 = time.time()
        qmu_a, qln_var_a = self.encode_a(x)
        encoding_time_1 = float(time.time() - encoding_time_1)

        a_enc = F.gaussian(qmu_a, qln_var_a)
        self.a_enc = a_enc 
        
        encoding_time_2 = time.time()
        qmu_z, qln_var_z = self.encode_z(x, a_enc)
        encoding_time_2 = float(time.time() - encoding_time_2)

        encoding_time = encoding_time_1 + encoding_time_2

        decoding_time_average = 0.

        self.kl = 0
        self.logp = 0

        logp_a_z = 0
        logp_x_az = 0
        logp_z = 0
        logq_a_x = 0
        logq_z_ax = 0

        current_temperature = min(self.temperature['value'],1.0)
        self.temperature['value'] += self.temperature['increment']

        for j in xrange(self.num_zsamples):
            # z ~ q(z|x, a)
            z = F.gaussian(self.qmu_z, self.qln_var_z)

            # Compute p(x|z)
            decoding_time = time.time()
            pmu_a, pln_var_a = self.decode_a(z)
            p_ber_prob_logit = self.decode(z)
            decoding_time = time.time() - decoding_time
            decoding_time_average += decoding_time

            logp_a_z += gaussian_logp(a_enc, pmu_a, pln_var_a)
            logp_x_az += bernoulli_logp(x, p_ber_prob_logit)
            logp_z += current_temperature*gaussian_logp0(z)
            logq_a_x += gaussian_logp(a_enc, qmu_a, qln_var_a)
            logq_z_ax += current_temperature*gaussian_logp(z, qmu_z, qln_var_z)

        logp_a_z /= self.num_zsamples
        logp_x_az /= self.num_zsamples
        logp_z /= self.num_zsamples
        logq_a_x /= self.num_zsamples
        logq_z_ax /= self.num_zsamples


        decoding_time_average /= self.num_zsamples
        self.logp /= self.num_zsamples

        self.obj_batch = logp_a_z + logp_x_az + logp_z - logq_a_x - logq_z_ax
        self.kl = logq_z_ax - logp_z
        self.logp = logp_x_az
        
        self.timing_info = np.array([encoding_time,decoding_time_average])

        batch_size = self.obj_batch.shape[0]
        
        self.obj = -F.sum(self.obj_batch)/batch_size
        
        return self.obj