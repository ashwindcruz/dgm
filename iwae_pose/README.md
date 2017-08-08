
# Importance Weighted Autoencoder (IWAE) Model for Human Pose

Implementation of the importance weighted autoencoder generative probabilistic model
as described in:

- Burda, Y., Grosse, R. and Salakhutdinov, R., 2015. 
	Importance weighted autoencoders. 
	[arXiv:1509.00519](https://arxiv.org/abs/1509.00519)

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
./train.py -g 0 -o vae1 -b 8192 -t 300 --nhidden 256 --time-print 10 \
    --time-sample 120 --vae-samples 10 --log-interval 10 --test 70000 ../data/MSRC12-X-d60.mat
```

Interpolating between random points (GPU id 0):

```
./interpolate.py -g 0 ../models/vae2 vae2_pathes.mat
```

