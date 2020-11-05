import sys
import time
import datetime
import re
import collections

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

import gpflow
import sklearn.cluster

from types import SimpleNamespace


def log_unhandled_exceptions(logger):
    # old_syshook = sys.excepthook
    def handler(*args):
        logger.critical("Uncaught exception", exc_info=args)
        # old_syshook(*args)

    sys.excepthook = handler


class TimeBlock:
    def __init__(self):
        self.time = None

    def __call__(self):
        return self.time

    def __enter__(self):
        self.time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.time = datetime.timedelta(seconds=time.perf_counter() - self.time)
        return exc_type is None


def plot_process(ax, x, p, d=2, color=None, label=None, zorder=None, **kwargs):
    if isinstance(p, pd.DataFrame):
        p = (p['mean'], p['variance'])
        assert len(p[0].shape) == 1 and len(p[1].shape) == 1
        p = (p[0].values.reshape(-1,1), p[1].values.reshape(-1,1))
        
    if len(x.shape) == 1:
        x = x.reshape(-1,1)
    
    if p[1].shape[1] == p[1].shape[0]:
        var_plot = ax.fill_between(
            x[:, 0],
            p[0][:, 0] - d * np.sqrt(np.diag(p[1])),
            p[0][:, 0] + d * np.sqrt(np.diag(p[1])),
            color=color,
            alpha=0.2,
            zorder=zorder
        )
    else:
        assert p[1].shape[1] == 1
        var_plot = ax.fill_between(
            x[:, 0],
            p[0][:, 0] - d * np.sqrt(p[1][:, 0]),
            p[0][:, 0] + d * np.sqrt(p[1][:, 0]),
            color=color,
            alpha=0.2,
            zorder=zorder
        )
    mean_plot = ax.plot(x, p[0], zorder=zorder, color=color, label=label, **kwargs)
    return mean_plot, var_plot


def legend_with_unique_labels(ax=None, f=None):
    if ax is None:
        ax = plt.gca()
    if f is None:
        f = ax
    label_to_handle = {l: h for h, l in zip(*ax.get_legend_handles_labels())}
    ax.legend(*zip(*((h, l) for l, h in label_to_handle.items())), loc="best")


class TrainTestSplit(collections.namedtuple("TrainTestSplit", ["train", "test"])):
    def replace(self, **kwargs):
        return self._replace(**kwargs)

    def sapply(self, f, *args, **kwargs):
        return TrainTestSplit(
            f(*self.train, *args, **kwargs), f(*self.test, *args, **kwargs)
        )

    def apply(self, f, *args, **kwargs):
        return TrainTestSplit(
            f(self.train, *args, **kwargs), f(self.test, *args, **kwargs)
        )

    def unzip(*args):
        return TrainTestSplit([a.train for a in args], [a.test for a in args])

    def zip(self):
        return (TrainTestSplit(*x) for x in zip(self.train, self.test))

    def from_sklearn(l):
        output = [None] * (len(l) // 2)
        for i in range(0, len(l), 2):
            output[i // 2] = TrainTestSplit(l[i], l[i + 1])
        return output


class IdentityScaler(object):
    def fit_transform(self, X, *args, **kwargs):
        return X

    def transform(self, X, *args, **kwargs):
        return X

    def fit(self, X, *args, **kwargs):
        return X

    def inverse_transform(self, X, *args, **kwargs):
        return X


def initial_inducing_points(X, m):
    if m < X.shape[0]:
        return (
            sklearn.cluster.KMeans(m, random_state=19960111, n_init=100)
            .fit(X)
            .cluster_centers_
        )
    else:
        return np.concatenate([X, np.random.randn(m - X.shape[0], X.shape[1])], 0)


def jitter(*args, **kwargs):
    return gpflow.settings.jitter * tf.eye(
        *args, **kwargs, dtype=gpflow.settings.float_type
    )


def cholesky_logdet(chol, name=None):
    return tf.multiply(
        tf.constant(2, dtype=gpflow.settings.float_type),
        tf.reduce_sum(tf.log(tf.linalg.diag_part(chol)), axis=-1),
        name=name,
    )


class obj(SimpleNamespace):
    def __getitem__(self, key):
        return self.__dict__[key]

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()


class _EmptySingleton(object):
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = object.__new__(cls)
        return cls.__instance

    def __repr__(self):
        return f"<singleton {repr(type(self).__name__)}>"


def EmptySingleton(name, module=None):
    name = sys.intern(str(name))
    if module is None:
        try:
            module = sys._getframe(1).f_globals.get("__name__", "__main__")
        except (AttributeError, ValueError):
            pass
    return type(name, (_EmptySingleton,), dict(__module__=module))


def to_tensor_debug(*tensors):
    params = ["{"]
    for tensor in tensors:
        params += ['"""', tensor.name, '""":"""', tensor, '""",']
    params += ["}"]
    return params


def from_tfprint_to_numpy(s):
    s = re.sub("(\d+(?:\.\d+)?)\s+", lambda m: f"{m.group(1)}, ", s)
    return np.array(eval(re.sub("\n ", ",", s)))


def from_tensor_debug(d):
    return {k.strip(): from_tfprint_to_numpy(v.strip()) for k, v in d.items()}


def print_tensors(*tensors):
    args = to_tensor_debug(*tensors)
    print_op = tf.print(
        "=begin=\n", *args, "\n\n", output_stream=sys.stdout, summarize=-1
    )
    return tf.control_dependencies([print_op])


def relative_error(a, b):
    a = np.array(a)
    b = np.array(b)
    assert a.dtype == b.dtype
    assert a.shape == b.shape

    abs_a = np.abs(a)
    abs_b = np.abs(b)
    diff = np.abs(a - b)
    min_normal = np.finfo(a.dtype).tiny
    max_value = np.finfo(a.dtype).max

    if np.equal(a, b):
        return 0
    elif (
        np.any(np.equal(a, 0))
        or np.any(np.equal(b, 0))
        or np.any(abs_a + abs_b <= min_normal)
    ):
        return diff * min_normal
    else:
        return diff / np.minimum(abs_a + abs_b, max_value)
