# Deep Generative Models
This repo contains implementations of several recent deep generative models.

### Auxiliary Deep Generative Models (ADGMs) and Skip Deep Generative Models (SDGMs)
- _Maaløe, L._, _Sønderby, C.K._, _Sønderby, S.K._ and _Winther, O._, 2016. Auxiliary deep generative models. 
arXiv preprint [arXiv:1602.05473](https://arxiv.org/abs/1602.05473).

### Householder Flow
- _Tomczak, J.M._ and _Welling, M._, 2016. Improving Variational Auto-Encoders using Householder Flow. 
arXiv preprint [arXiv:1611.09630](https://arxiv.org/abs/1611.09630).

### Importance Weighted Autoencoder (IWAE)
- _Burda, Y._, _Grosse, R._ and _Salakhutdinov, R._, 2015. Importance weighted autoencoders. 
arXiv preprint [arXiv:1509.00519](https://arxiv.org/abs/1509.00519).

### Inverse Autoregressive Flow (IAF)
- _Kingma, D.P._, _Salimans, T._ and _Welling, M._, 2016. Improving variational inference with inverse autoregressive flow. 
arXiv preprint [arXiv:1606.04934](https://arxiv.org/abs/1606.04934).

### Normalizing Flows
- _Rezende, D.J._ and _Mohamed, S._, 2015. Variational inference with normalizing flows. 
arXiv preprint [arXiv:1505.05770](https://arxiv.org/abs/1505.05770).

### Variational Autoencoder (VAE) 
- _Diederik P. Kingma_, and _Max Welling_, 2013. Auto-encoding variational bayes.
  arXiv preprint [arXiv:1312.6114](http://arxiv.org/abs/1312.6114).
- _Danilo Jimenez Rezende_, _Shakir Mohamed_, and _Daan Wierstra_, 2014. Stochastic
  backpropagation and approximate inference in deep generative models.
  arXiv preprint [arXiv:1401.4082](http://arxiv.org/abs/1401.4082).

## Data
The data used for experimentation is not supplied in this repo at this time but can be made available upon requests.

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
/train.py -g 0 -o 'demo_model' --model-type vae --vae-samples 1 --ntrans 1 --nlatent 16 --nhidden 512 --nlayers 4\
 -b 16384 --batch-limit 1000 -t 1000000 --time-print 600 --epoch-sample 100 --log-interval 5 --data pose\
 --init-temp 0 --temp-epoch 200  --init-learn 1e-4 --learn-decay 3e-3 --weight-decay 0 --init-model none 
```



