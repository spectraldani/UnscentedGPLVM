# Unscented Bayesian GPLVM

This repository contains all the code needed to replicate the experiments presented
in the article "Unscented Bayesian GPLVM", preprint avaible at [arXiv][arxiv]. The code for
the Unscented Bayesian GPLVM model is not neatly packaged into a Python package but
can be readly imported and used. See any of the Jupyter notebooks for example usage.

## Dependencies

|  Dependency  | Version |
|:------------:|:-------:|
| Python       | 3.6.6   |
| GPFlow       | 1.3.0   |
| Tensorflow   | 1.11.0  |
| Numpy        | 1.15.2  |
| Pandas       | 0.23.4  |
| Scikit-Learn | 0.20.0  |

## How to replicate the experiments

Each Jupyter notebook have two variables named `dataset` and `save_or_load`,
these variables control which dataset is being used and what the notebook should
do. They are located in the third cell of each notebook.

### `dataset` variable
For the dimensionality reduction task we used the following datasets:

|   `dataset` value   |         Dataset         |
|:-------------------:|:-----------------------:|
| `"oil flow"`        | Three Phase Oil Dataset |
| `"USPS digits"`     | USPS Digits from 0 to 4 |
| `"USPS digits all"` | USPS Digits from 0 to 9 |

For the free simulation we used the following datasets:

| `dataset` value |              Dataset             |
|:---------------:|:--------------------------------:|
| `"passengers"`  | International Airline Passengers |

### `save_or_load` variable

On both notebooks, the `save_or_load` variable controls the following behavior:

| `save_or_load` value |                               Behaviour                              |
|:--------------------:|:--------------------------------------------------------------------:|
| `"save"`             | Run experiments and save images, tables and latent space/predictions |
| `"load"`             | Load latent space/predictions and show images/tables on Notebook     |
| `"loadsave"`         | Load latent space/predictions and resave images and tables           |
| `"rerun"`            | Run experiments but don't save any data                              |

<!-- Todo: Insert preprint link here -->
[arxiv]: about:blank
