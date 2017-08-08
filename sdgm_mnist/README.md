
# Variational Autoencoder (VAE) Model for Human Pose

Implementation of the variational autoencoder generative probabilistic model
as described in:

- _Diederik P. Kingma_, and _Max Welling_. "Auto-encoding variational bayes."
  arXiv preprint [arXiv:1312.6114](http://arxiv.org/abs/1312.6114) (2013).
- _Danilo Jimenez Rezende_, _Shakir Mohamed_, and _Daan Wierstra_. "Stochastic
  backpropagation and approximate inference in deep generative models."
  arXiv preprint [arXiv:1401.4082](http://arxiv.org/abs/1401.4082) (2014).

## Demo

![Sampling VAE latent space](models/vae2_path_2.gif)

The above animation shows a linear interpolation in latent _z_-space between
two random points.

## Data

We include human pose data from the following data sets:

- [MSRC-12 Kinect gesture data set](http://research.microsoft.com/en-us/um/cambridge/projects/msrc12/)
  These are 702,551 frames of 60 dimensions, describing 20 joints in XYZ world
  space.  Included at `data/MSRC12-X-d60.mat`.

## Requirements

The following Python packages are required (Install using `pip install --user <package>`):

- `pyyaml`
- `h5py`
- `scipy`
- `docopt`
- `chainer`

## Training

Example training run (GPU id 0):

```
./train.py -g 0 -o vae1 -b 8192 --batch-limit 1000 -t 300 --nhidden 256 --time-print 10 \
    --time-sample 1200 --vae-samples 1 --log-interval 10 --test 70000 ../data/MSRC12-X-d60.mat
```

Interpolating between random points (GPU id 0):

```
./interpolate.py -g 0 ../models/vae2 vae2_pathes.mat
```

