import pdb

import numpy as np
import math
import time

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda

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

        # flow
        for i in range(num_trans):
            layer_name = 'flow_w_' + str(i) # weights
            setattr(self, layer_name, L.Scale(axis=1, W_shape=(dim_latent), bias_term=False))  
            self._children.append(layer_name)

            layer_name = 'flow_b_' + str(i) # bias
            setattr(self, layer_name, L.Bias(axis=0, shape=(1)))  
            self._children.append(layer_name)

            layer_name = 'flow_u_' + str(i) # scaling factor u
            setattr(self, layer_name, L.Scale(axis=1, W_shape=(dim_latent), bias_term=False))  
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
        
       return self.qmu, self.qln_var

    def decode(self, z):
        h = F.crelu(self.plin0(z))

        for i in range(self.num_layers-1):
            layer_name = 'plin' + str(i+1)
            h = F.crelu(self[layer_name](h))

        self.p_ber_prob_logit = self.plin_ber_prob(h)

        return self.p_ber_prob_logit
    
    def planar_flows(self,z):
        self.z_trans = []
        self.z_trans.append(z)
        self.phi = []

        for i in range(self.num_trans):
            flow_w_name = 'flow_w_' + str(i)
            flow_b_name = 'flow_b_' + str(i)
            flow_u_name = 'flow_u_' + str(i)

            h = self[flow_w_name](z)
            h = F.sum(h,axis=(1))
            h = self[flow_b_name](h)
            h = F.tanh(h)
            h_tanh = h

            dim_latent = z.shape[1]
            h = F.transpose(F.tile(h, (dim_latent,1)))
            h = self[flow_u_name](h)

            z += h

            self.z_trans.append(z)

            # Calculate and store the phi term
            h_tanh_derivative = 1-(h_tanh*h_tanh)
            h_tanh_derivative = F.transpose(F.tile(h_tanh_derivative, (dim_latent,1))) 
            
            phi = self[flow_w_name](h_tanh_derivative) # Equation (11)
            self.phi.append(phi)

        return z


    def __call__(self, x):
        # Compute q(z|x)
        encoding_time = time.time()
        qmu, qln_var = self.encode(x)
        encoding_time = float(time.time() - encoding_time)

        decoding_time_average = 0.

        self.kl = 0
        self.logp = 0

        current_temperature = min(self.temperature['value'],1.0)
        self.temperature['value'] += self.temperature['increment']

        for j in xrange(self.num_zsamples):
            # Sample z ~ q(z_0|x)
            z_0 = F.gaussian(self.qmu, self.qln_var)

            # Perform planar flow mappings, Equation (10)
            decoding_time = time.time()
            z_K = self.planar_flows(z_0)

            # Obtain parameters for p(x|z_K)
            p_ber_prob_logit =  self.decode(z_K)
            decoding_time = time.time() - decoding_time
            decoding_time_average += decoding_time

            # Compute log q(z_0)
            q_prior_log = current_temperature*gaussian_logp0(z_0)
            
            # Compute log p(x|z_K)
            decoder_log = bernoulli_logp(x, p_ber_prob_logit)
            # Compute log p(z_K)
            p_prior_log = current_temperature*gaussian_logp0(z_K)

            # Compute log p(x,z_K) which is log p(x|z_K) + log p(z_K)
            joint_log = decoder_log + p_prior_log

            # Compute second term of log q(z_K)
            q_K_log = 0
            for i in range(self.num_trans):
                flow_u_name = 'flow_u_' + str(i)
                lodget_jacobian = F.sum(self[flow_u_name](self.phi[i]), axis=1)
                q_K_log += F.log(1 + lodget_jacobian)
            q_K_log *= current_temperature

            # For recording purposes only
            self.logp += decoder_log
            self.kl += -(q_prior_log - p_prior_log - q_K_log)


        decoding_time_average /= self.num_zsamples
        # pdb.set_trace()
        self.obj_batch = ((q_prior_log -joint_log) - q_K_log)
        self.obj_batch /= self.num_zsamples
        batch_size = self.obj_batch.shape[0]

        self.obj = F.sum(self.obj_batch)/batch_size

        self.kl /= self.num_zsamples
        self.logp /= self.num_zsamples

        self.timing_info = np.array([encoding_time,decoding_time])
        
        return self.obj

