# Unscented Bayesian GPLVM

This repository contains all the code needed to replicate the experiments
presented in the article "Learning GPLVM with arbitrary kernels using the
unscented transformation", preprint avaible at [arXiv][arxiv]. The code for the
Unscented Bayesian GPLVM model is not neatly packaged into a Python package yet
but can be readly imported and used. See any of the Jupyter notebooks for
example usage.

As noted, part of the code was adapted from the [GPFlow](https://github.com/GPflow/GPflow/tree/v1.5.1/gpflow)
project.

## Dependencies
See [requirements.txt](./requirements.txt).

## How to replicate the experiments

Each Jupyter notebook have two variables named `dataset` and `save_or_load`,
these variables control which dataset is being used and what the notebook should
do. They are located in the third cell of each notebook.

### `dataset` variable
For the dimensionality reduction task we used the following datasets:

|   `dataset` value   |         Dataset         |
|:-------------------:|:-----------------------:|
| `"oil flow"`        | Three Phase Oil dataset |
| `"USPS digits"`     | USPS Digits dataset     |

For the free simulation we used the following datasets:

| `dataset` value |              Dataset             |
|:---------------:|:--------------------------------:|
| `"passengers"`  | International Airline Passengers |

### `save_or_load` variable

On both notebooks, the `save_or_load` variable controls the following behavior:

| `save_or_load` value |                               Behaviour                              |
|:--------------------:|:--------------------------------------------------------------------:|
| `"save"`             | Run experiments and save images, tables and latent space/predictions |
| `"rerun"`            | Run experiments but don't save any data                              |


[arxiv]: https://arxiv.org/abs/1907.01867
