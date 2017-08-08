#!/usr/bin/env python

"""Train variational autoencoder (VAE) model for pose data.

Author: Sebastian Nowozin <senowozi@microsoft.com>
Date: 5th August 2016

Usage:
  train.py (-h | --help)
  train.py [options] <data.mat>

Options:
  -h --help                     Show this help screen.
  -g <device>, --device         GPU id to train model on.  Use -1 for CPU [default: -1].
  -o <modelprefix>              Write trained model to given file.h5 [default: output].
  --vis <graph.ext>             Visualize computation graph.
  -b <batchsize>, --batchsize   Minibatch size [default: 100].
  -t <runtime>, --runtime       Total training runtime in seconds [default: 7200].
  --vae-samples <zcount>        Number of samples in VAE z [default: 1]
  --nhidden <nhidden>           Number of hidden dimensions [default: 128].
  --nlatent <nz>                Number of latent VAE dimensions [default: 16].
  --time-print=<sec>            Print status every so often [default: 60].
  --time-sample=<sec>           Print status every so often [default: 600].
  --dump-every=<sec>            Dump model every so often [default: 900].
  --log-interval <log-interval> Number of batches before logging training and testing ELBO [default: 100].
  --test <test>                 Number of samples to set aside for testing [default:70000]
  --nmap <nmap>                 Number of planar flow mappings to apply [default:1]

The data.mat file must contain a (N,d) array of N instances, d dimensions
each.
"""

import time
import yaml
import numpy as np
import h5py
import scipy.io as sio
from docopt import docopt

import chainer
from chainer import serializers
from chainer import optimizers
from chainer import cuda
from chainer import computational_graph
import chainer.functions as F
import cupy

import model
import util

import pdb

args = docopt(__doc__, version='train 0.1')
print(args)

print "Using chainer version %s" % chainer.__version__

# Loading training data
data_mat = h5py.File(args['<data.mat>'], 'r')
X = data_mat.get('X')
X = np.array(X)
X = X.transpose()
N = X.shape[0]
d = X.shape[1]
print "%d instances, %d dimensions" % (N, d)

# Split data into training and testing data
#X  = np.random.permutation(X) # To make things easier for debugging, split testing and training without mixing up indicees that we use
test_size = int(args['--test'])
X_test = X[0:test_size,:]
X_train = X[test_size:,:]

N = X_train.shape[0]
#N -= test_size

# Set up model
nhidden = int(args['--nhidden'])
print "%d hidden dimensions" % nhidden
nlatent = int(args['--nlatent'])
print "%d latent VAE dimensions" % nlatent
zcount = int(args['--vae-samples'])
print "Using %d VAE samples per instance" % zcount
nmap = int(args['--nmap'])
print "Using %d planar flow mappings" % nmap

log_interval = int(args['--log-interval'])
print "Recording training and testing ELBO every %d batches" % log_interval

# Setup training parameters
batchsize = int(args['--batchsize'])
print "Using a batchsize of %d instances" % batchsize

vae = model.VAE(d, nhidden, nlatent, zcount, nmap)
opt = optimizers.Adam()
opt.setup(vae)
opt.add_hook(chainer.optimizer.GradientClipping(4.0))

# Move to GPU
gpu_id = int(args['--device'])
if gpu_id >= 0:
    cuda.check_cuda_available() # comment out to surpress an unncessarry warning
if gpu_id >= 0:
    xp = cuda.cupy
    vae.to_gpu(gpu_id)
else:
    xp = np

start_at = time.time()
period_start_at = start_at
period_bi = 0
runtime = int(args['--runtime'])

print_every_s = float(args['--time-print'])
print_at = start_at + print_every_s

sample_every_s = float(args['--time-sample'])
sample_at = start_at + sample_every_s

bi = 0  # batch index
printcount = 0

obj_mean = 0.0
obj_count = 0

with cupy.cuda.Device(gpu_id):
    xp.random.seed(0)
    # Set up variables that cover the entire training and testing sets
    x_train = chainer.Variable(xp.asarray(X_train, dtype=np.float32))
    x_test = chainer.Variable(xp.asarray(X_test, dtype=np.float32))
    
    # Set up the training and testing log files
    train_log_file = args['-o'] + '_train_log.txt'
    test_log_file  = args['-o'] +  '_test_log.txt' 

    with open(train_log_file, 'w+') as f:
        f.write('Training Log \n')

    with open(test_log_file, 'w+') as f:
        f.write('Testing Log \n')

    while True:
        bi += 1
        period_bi += 1

        now = time.time()
        tpassed = now - start_at

        # Check whether we exceeded training time
        if tpassed >= runtime:
            print "Training time of %ds reached, training finished." % runtime
            break

        total = bi * batchsize

        # Print status information
        if now >= print_at:
        #if True:
            print_at = now + print_every_s
            printcount += 1
            tput = float(period_bi * batchsize) / (now - period_start_at)
            EO = obj_mean / obj_count
            print "   %.1fs of %.1fs  [%d] batch %d, E[obj] %.4f,  %.2f S/s, %d total" % \
                  (tpassed, runtime, printcount, bi, EO, tput, total)

            period_start_at = now
            obj_mean = 0.0
            obj_count = 0
            period_bi = 0

        vae.zerograds()

        # Build training batch (random sampling without replacement)
        J = np.sort(np.random.choice(N, batchsize, replace=False))
        x = chainer.Variable(xp.asarray(X_train[J,:], dtype=np.float32))

        obj = vae(x)
        obj_mean += obj.data
        obj_count += 1

        # (Optionally:) visualize computation graph
        if bi == 1 and args['--vis'] is not None:
            print "Writing computation graph to '%s'." % args['--vis']
            g = computational_graph.build_computational_graph([obj])
            util.print_compute_graph(args['--vis'], g)

        # Update model parameters
        obj.backward()
        opt.update()

        # Sample a set of poses
        if now >= sample_at:
            sample_at = now + sample_every_s
            print "   # sampling"
            z = np.random.normal(loc=0.0, scale=1.0, size=(1024,nlatent))
            z = chainer.Variable(xp.asarray(z, dtype=np.float32))
            vae.decode(z)
            Xsample = F.gaussian(vae.pmu, vae.pln_var)
            Xsample.to_cpu()
            sio.savemat('%s_samples_%d.mat' % (args['-o'], total), { 'X': Xsample.data })

        # Get the ELBO for the training and testing set and record it
        # -1 is because we want to record the first set which has bi value of 1
        if((bi-1)%log_interval==0):
                        
            whole_batch_size = 8192
            
            # Training results
            training_obj = 0
            for i in range(0,N/whole_batch_size):
                x_train = chainer.Variable(xp.asarray(X_train[i*whole_batch_size:(i+1)*whole_batch_size,:], dtype=np.float32))
                obj = vae(x_train)
                training_obj += -obj.data
            # One final smaller batch to cover what couldn't be captured in the loop
            #x_train = chainer.Variable(xp.asarray(X_train[(N/whole_batch_size)*whole_batch_size:,:], dtype=np.float32))
            #obj_train = vae(x_train)
            #training_obj += -obj_train.data
            
            training_obj /= ((N/whole_batch_size)-1) # We want to average by the number of batches
            with open(train_log_file, 'a') as f:
                f.write(str(training_obj) + '\n')
            
            vae.cleargrads()

            # Testing results
            #testing_obj = 0
            #for i in range(0,N/whole_batch_size):
            #    x_test = chainer.Variable(xp.asarray(X_test[i*whole_batch_size:(i+1)*whole_batch_size,:], dtype=np.float32))
            #    obj = vae(x_test)
            #    testing_obj += -obj.data
            # One final smaller batch to cover what couldn't be captured in the loop
            #x_test = chainer.Variable(xp.asarray(X_test[(N/whole_batch_size)*whole_batch_size:,:], dtype=np.float32))
            #obj_test = vae(x_test)
            #testing_obj = -obj_test.data
            
            #testing_obj /= (N/whole_batch_size) # We want to average by the number of batches
            #with open(train_log_file, 'a') as f:
            #    f.write(str(testing_obj) + '\n')
            
            #vae.cleargrads()

# Save model
if args['-o'] is not None:
    modelmeta = args['-o'] + '.meta.yaml'
    print "Writing model metadata to '%s' ..." % modelmeta
    with open(modelmeta, 'w') as outfile:
        outfile.write(yaml.dump(dict(args), default_flow_style=False))

    modelfile = args['-o'] + '.h5'
    print "Writing model to '%s' ..." % modelfile
    serializers.save_hdf5(modelfile, vae)

