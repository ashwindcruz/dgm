#!/usr/bin/env python

"""Train variational autoencoder (VAE) model for pose data.

Author: Sebastian Nowozin <senowozi@microsoft.com>, Ashwin D'Cruz <ashwindcruz94@gmail.com>
Date: 5th August 2016

Usage:
  train.py (-h | --help)
  train.py [options]

Options:
  -h --help                     Show this help screen.
  -g <device>, --device         GPU id to train model on.  Use -1 for CPU [default: -1].
  --model-type <model-type>     Type of model to use (examples: vae, iwae, householder, planar) [default: vae].
  -o <model-prefix>             Write trained model to given file.h5 [default: output].
  --vis <graph.ext>             Visualize computation graph.
  -b <batchsize>, --batchsize   Minibatch size [default: 100].
  --batch-limit <batch-limit>   Total number of batches to process for training [default: -1]
  -t <runtime>, --runtime       Total training runtime in seconds [default: 7200].
  --nhidden <nhidden>           Number of hidden dimensions [default: 256].
  --nlatent <nz>                Number of latent VAE dimensions [default: 16].
  --nlayers <nl>                Number of hidden layers in the encoder and decoder network [default: 4]
  --time-print=<sec>            Print status every so often [default: 60].
  --epoch-sample=<sec>          Sample every so often [default: 10].
  --dump-every=<sec>            Dump model every so often [default: 900].
  --log-interval <log-interval> Number of batches before logging training and testing ELBO [default: 100].
  --vae-samples <zcount>        Number of samples in VAE/IWAE z [default: 1].
  --ntrans <ntrans>             Number of Householder flow transformations to apply. Only applicable with householder and planar model [default: 2]. 
  --data <data>                 Prefix of mat files that will be used for training and testing. 
  --init-temp <init-temp>       Initial KL temperature [default: 0].
  --temp-epoch <temp-epoch>     Number of epochs used to increase temperature from init-temp to 1 [default: 200].
  --init-learn <init-learn>     Initial learning rate [default: 1e-3].
  --learn-decay <learn-decay>   Learning rate decay [default: 1e-3].
  --weight-decay <weight-decay> Weight decay value for regularization [default: 0].
  --init-model <init-model>     Initialize using a pre-trained model [default: none]

The data.mat file must contain a (N,d) array of N instances, d dimensions
each.
"""

import math
import time
import yaml
import numpy as np
import h5py
import scipy.io as sio
from docopt import docopt
import os

import chainer
from chainer import serializers
from chainer import optimizers
from chainer import cuda
from chainer import computational_graph
import chainer.functions as F
import cupy

from adgm_pose import model as adgm_pose
from iaf_pose import model as iaf_pose 
from iwae_pose import model as iwae_pose 
from householder_pose import model as householder_pose
from planar_pose import model as planar_pose 
from sdgm_pose import model as sdgm_pose
from vae_pose import model as vae_pose

from adgm_mnist import model as adgm_mnist
from iaf_mnist import model as iaf_mnist
from iwae_mnist import model as iwae_mnist
from householder_mnist import model as householder_mnist
from planar_mnist import model as planar_mnist
from sdgm_mnist import model as sdgm_mnist
from vae_mnist import model as vae_mnist

from adgm_celebA import model as adgm_celebA
from iwae_celebA import model as iwae_celebA
from householder_celebA import model as householder_celebA
from vae_celebA import model as vae_celebA
from sdgm_celebA import model as sdgm_celebA

import util

import pdb

args = docopt(__doc__, version='train 0.1')
print(args)

print "Using chainer version %s" % chainer.__version__

# Loading training and validation data
if(args['--data']!='celebA_scaled'):
  data_mat = sio.loadmat('./data/' + args['--data'] + '_training.mat')
  X_train = data_mat.get('X')
  N = X_train.shape[0]
  d = X_train.shape[1] 
  print "%d instances, %d dimensions" % (N, d)

  data_mat = sio.loadmat('./data/' + args['--data'] + '_validation.mat')
  X_validation = data_mat.get('X')
else:
  h5f = h5py.File('./data/celebA_scaled_training.h5', 'r')
  X_train = h5f['X'][:]
  N = X_train.shape[0]
  d = X_train.shape[1]
  h5f.close()
  print "%d instances, %d dimensions" % (N, d)

  h5f = h5py.File('./data/celebA_scaled_validation.h5', 'r')
  X_validation = h5f['X'][:]
  h5f.close()

# Set up model
nhidden = int(args['--nhidden'])
print "%d hidden dimensions" % nhidden
nlatent = int(args['--nlatent'])
print "%d latent VAE dimensions" % nlatent
nlayers = int(args['--nlayers'])
print "%d hidden layers" % nlayers
zcount = int(args['--vae-samples'])
print "Using %d VAE samples per instance" % zcount

log_interval = int(args['--log-interval'])
print "Recording training and testing ELBO every %d batches" % log_interval

# Provide initial temperature and number of epochs for the schedule
temperature = {}
temperature['value'] = float(args['--init-temp'])
temperature_epochs = float(args['--temp-epoch'])
temperature['increment'] = (1.0-temperature['value'])/temperature_epochs

# Check which model was specified
model_type = args['--model-type']
data_type = args['--data']

ntrans = int(args['--ntrans'])

if data_type=='pose':
  if model_type=='vae':
    vae = vae_pose.VAE(d, nhidden, nlatent, nlayers, temperature, zcount)
  elif model_type=='adgm':
    vae = adgm_pose.VAE(d, nhidden, nlatent, nlayers, temperature, zcount)  
  elif model_type=='sdgm':
    vae = sdgm_pose.VAE(d, nhidden, nlatent, nlayers, temperature, zcount)  
  elif model_type=='iwae':
    vae = iwae_pose.VAE(d, nhidden, nlatent, nlayers, temperature, zcount)
  elif model_type=='householder':
    print 'Using %d Householder flow transformations' % ntrans
    vae = householder_pose.VAE(d, nhidden, nlatent, nlayers, ntrans, temperature, zcount)
  elif model_type=='planar':
    print 'Using %d Planar flow transformations' % ntrans
    vae = planar_pose.VAE(d, nhidden, nlatent, nlayers, ntrans, temperature, zcount)
  elif model_type=='iaf':
    print 'Using %d IAF transformations' % ntrans
    vae = iaf_pose.VAE(d, nhidden, nlatent, nlayers, ntrans, temperature, zcount)
elif data_type=='mnist':
  if model_type=='vae':
    vae = vae_mnist.VAE(d, nhidden, nlatent, nlayers, temperature, zcount)
  elif model_type=='adgm':
    vae = adgm_mnist.VAE(d, nhidden, nlatent, nlayers, temperature, zcount) 
  elif model_type=='sdgm':
    vae = sdgm_mnist.VAE(d, nhidden, nlatent, nlayers, temperature, zcount)  
  elif model_type=='iwae':
    vae = iwae_mnist.VAE(d, nhidden, nlatent, nlayers, temperature, zcount)
  elif model_type=='householder':
    print 'Using %d Householder flow transformations' % ntrans
    vae = householder_mnist.VAE(d, nhidden, nlatent, nlayers, ntrans, temperature, zcount)
  elif model_type=='planar':
    print 'Using %d Planar flow transformations' % ntrans
    vae = planar_mnist.VAE(d, nhidden, nlatent, nlayers, ntrans, temperature, zcount)
  elif model_type=='iaf':
    print 'Using %d IAF transformations' % ntrans
    vae = iaf_mnist.VAE(d, nhidden, nlatent, nlayers, ntrans, temperature, zcount)
elif data_type=='celebA_scaled':
  if model_type=='vae':
    vae = vae_celebA.VAE(d, nhidden, nlatent, nlayers, temperature, zcount)
  elif model_type=='adgm':
    vae = adgm_celebA.VAE(d, nhidden, nlatent, nlayers, temperature, zcount)
  elif model_type=='sdgm':
    vae = sdgm_celebA.VAE(d, nhidden, nlatent, nlayers, temperature, zcount)
  elif model_type=='iwae':
    vae = iwae_celebA.VAE(d, nhidden, nlatent, nlayers, temperature, zcount)
  elif model_type=='householder':
    print 'Using %d Householder flow transformations' % ntrans
    vae = householder_celebA.VAE(d, nhidden, nlatent, nlayers, ntrans, temperature, zcount)
# Load in pre trained model if provided
init_model = args['--init-model']
if(init_model != 'none'):
    serializers.load_hdf5(init_model, vae)
    with h5py.File(init_model,'r') as f:
      vae.epochs_seen = f['epochs_seen'].value
      vae.temperature['value'] = f['temperature_value'].value 
      vae.temperature['increment'] = f['temperature_increment'].value
    # pdb.set_trace()
# Set up learning rate parameters. Specifically, there is an exponential decay on the learning rate and the parameters for those are set here.
alpha_0 = float(args['--init-learn'])
k_decay = float(args['--learn-decay'])

opt = optimizers.Adam(alpha=alpha_0)

opt.setup(vae)
opt.add_hook(chainer.optimizer.GradientClipping(4.0))
opt.add_hook(chainer.optimizer.WeightDecay(float(args['--weight-decay'])))

opt.alpha = alpha_0*math.exp(-k_decay*(vae.epochs_seen-1))

# Move to GPU
gpu_id = int(args['--device'])
if gpu_id >= 0:
    cuda.check_cuda_available()
if gpu_id >= 0:
    xp = cuda.cupy
    vae.to_gpu(gpu_id)
else:
    xp = np

# Setup training parameters
batch_size = int(args['--batchsize'])
print "Using a batchsize of %d instances" % batch_size
batch_limit = int(args['--batch-limit'])
if batch_limit!=-1:
    print "Limiting training to run for %d epochs" % batch_limit

start_at = time.time()
period_start_at = start_at
period_bi = 0
runtime = int(args['--runtime'])

print_every_s = float(args['--time-print'])
print_at = start_at + print_every_s

sample_every_epoch = float(args['--epoch-sample'])

bi = 0  # epoch index
printcount = 0

obj_mean = 0.0
obj_count = 0

# Sample counter
counter = 0

# Folder where results will be saved
directory = model_type + '_' + args['-o'] + '_results'
if not os.path.exists(directory):
    os.makedirs(directory)

with cupy.cuda.Device(gpu_id):
    
    # Set up the training and testing log files
    online_log_file = directory + '/'  + 'online_log.txt' 
    train_log_file = directory  + '/'  + 'train_log.txt'
    test_log_file  = directory  + '/'  +  'test_log.txt' 

    with open(online_log_file, 'w+') as f:
        f.write('ELBO, KL, Logp, SEM, Encoder Time, Decoder Time, Backward Time \n')

    with open(train_log_file, 'w+') as f:
        f.write('ELBO, KL, Logp, SEM, Encoder Time, Decoder Time, Backward Time \n')

    with open(test_log_file, 'w+') as f:
        f.write('ELBO, KL, Logp, SEM, Encoder Time, Decoder Time, Backward Time \n')
    # pdb.set_trace()
    while True:
        bi += 1
        period_bi += 1

        now = time.time()
        tpassed = now - start_at

        # Check whether we exceeded training time
        if tpassed >= runtime:
            print "Training time of %ds reached, training finished." % runtime
            break

        # Check whether we exceeded the batch limit
        if bi > batch_limit:
            print " Batch limit of %d reached, training finished." % batch_limit
            break

        total = bi * batch_size

        # Print status information
        if now >= print_at:
        #if True:
            print_at = now + print_every_s
            printcount += 1
            tput = float(period_bi * batch_size) / (now - period_start_at)
            if(obj_count==0):
                obj_count+=1
            EO = -obj_mean / obj_count
            print "   %.1fs of %.1fs  [%d] epoch %d, ELBO %.4f, KL %.4f, Logp %.4f,  %.2f S/s, %d total" % \
                  (tpassed, runtime, printcount, (bi-1), EO, xp.mean(vae.kl.data), xp.mean(vae.logp.data), tput, total)

            period_start_at = now
            obj_mean = 0.0
            obj_count = 0
            period_bi = 0

        X_online = np.random.permutation(X_train)
        # X_online = xp.asarray(X_online_np, dtype=np.float32)
        
        # if(bi>20):
        #   pdb.set_trace()
        vae = util.evaluate_dataset(vae, X_online, batch_size, online_log_file, True, opt)
                
        # If the model breaks, skip this training iteration
        if(math.isnan(vae.obj.data)):
            bi -=1
            continue
            
        obj_mean += vae.obj.data
        obj_count += 1
        vae.epochs_seen += 1
        # pdb.set_trace()
        opt.alpha = alpha_0*math.exp(-k_decay*vae.epochs_seen)

        # pdb.set_trace()
        # Get the ELBO for the training and testing set and record it
        # -1 is because we want to record the first set which has bi value of 1
        if((bi-1)%log_interval==0):
            #print('##################### Post Epoch Evaluation      #####################')
            #util.evaluate_dataset(vae, X_train, batch_size, train_log_file, False, opt)
            util.evaluate_dataset(vae, X_validation, batch_size, test_log_file, False, opt)   

            if ((args['-o'] is not None) and ((bi-1)%(log_interval*100)==0)): #Additional *100 term because we don't want a checkpoint every log point
                print('##################### Saving Model Checkpoint     #####################')

                batch_number = str(bi).zfill(6)
                modelfile = directory + '/' + batch_number + '.h5'
                print "Writing model checkpoint to '%s' ..." % (modelfile)
                serializers.save_hdf5(modelfile, vae)

        # (Optionally:) visualize computation graph
        if bi == 1 and args['--vis'] is not None:
            print "Writing computation graph to '%s/%s'." % (directory,args['--vis'])
            g = computational_graph.build_computational_graph([obj])
            util.print_compute_graph(directory + '/' + args['--vis'], g)

        # Sample a set of poses
        if (bi%sample_every_epoch==0) and data_type=='pose':
            counter +=1
            print "   # sampling"
            z = np.random.normal(loc=0.0, scale=1.0, size=(1024,nlatent))
            z = chainer.Variable(xp.asarray(z, dtype=np.float32), volatile='ON')
            vae.decode(z)
            Xsample = F.gaussian(vae.pmu, vae.pln_var)
            Xsample.to_cpu()
            sio.savemat('%s/samples_%d.mat' % (directory, counter), { 'X': Xsample.data })
            vae.pmu.to_cpu()
            sio.savemat('%s/means_%d.mat' % (directory, counter), { 'X': vae.pmu.data })
        elif(bi%sample_every_epoch==0) and (data_type=='mnist' or data_type=='celebA_scaled'):
            counter +=1
            print "   # sampling"
            z = np.random.normal(loc=0.0, scale=1.0, size=(8,nlatent))
            z = chainer.Variable(xp.asarray(z, dtype=np.float32), volatile='ON')
            vae.decode(z)
            Xsample_ber_prob = (F.sigmoid(vae.p_ber_prob_logit))
            Xsample_ber_prob.to_cpu()
            Xsample_ber_prob = Xsample_ber_prob.data
            Xsample = Xsample_ber_prob.data # For celebA

            if(data_type=='mnist'):
              Xsample = np.random.binomial(1, p=Xsample_ber_prob)
            
            # Xsample.to_cpu()
            sio.savemat('%s/samples_%d.mat' % (directory, counter), { 'X': Xsample})
            
# Record final information 

util.evaluate_dataset(vae, X_train, batch_size, train_log_file, False, opt)
util.evaluate_dataset(vae, X_validation, batch_size, test_log_file, False, opt)  

#counter +=1
#print "   # sampling"
#z = np.random.normal(loc=0.0, scale=1.0, size=(1024,nlatent))
#z = chainer.Variable(xp.asarray(z, dtype=np.float32))
#vae.decode(z)
#Xsample = F.gaussian(vae.pmu, vae.pln_var)
#Xsample.to_cpu()
#sio.savemat('%s/samples_%d.mat' % (directory, counter), { 'X': Xsample.data })
#vae.pmu.to_cpu()
#sio.savemat('%s/means_%d.mat' % (directory, counter), { 'X': vae.pmu.data })

# Save model
if args['-o'] is not None:
    modelmeta = directory + '/meta.yaml'
    print "Writing model metadata to '%s' ..." % (modelmeta)
    with open(modelmeta, 'w') as outfile:
        outfile.write(yaml.dump(dict(args), default_flow_style=False))

    modelfile = directory + '/' + args['-o'] + '.h5'
    print "Writing final model to '%s' ..." % (modelfile)
    serializers.save_hdf5(modelfile, vae)
    with h5py.File(modelfile,'a') as f:
      f['epochs_seen'] = vae.epochs_seen
      f['temperature_increment'] = vae.temperature['increment']
      f['temperature_value'] = vae.temperature['value']
