import gpflow
import numpy as np
import tensorflow as tf

float_type = gpflow.settings.float_type


def KxzKzx(k1, feat1, k2, feat2):
    def inner(x):
        # n * samples * m1
        K1 = k1.K(x, feat1.Z)
        # n * samples * m2
        K2 = k2.K(x, feat2.Z)

        # n * samples * m1 * m2
        return K1[:, :, :, None] * K2[:, :, None, :]

    return inner


def is_feature(o):
    return isinstance(o, gpflow.features.InducingFeature)


def is_kernel(o):
    return isinstance(o, gpflow.kernels.Kernel)


class Expectation:
    def __init__(self):
        self.p = None

    def load(self, p):
        self.p = p

    def expectation(self, obj1, feat1, obj2, feat2):
        raise NotImplementedError()


class AnalyticExpectation(Expectation):
    @property
    def __name__(self):
        return f"analytic"

    def expectation(self, obj1, feat1, obj2, feat2):
        return gpflow.expectations._expectation(self.p, obj1, feat1, obj2, feat2, nghp=0)


class QuadratureExpectation(Expectation):
    def quadrature(self, f):
        raise NotImplementedError()

    def expectation(self, obj1, feat1, obj2, feat2):
        assert isinstance(
            self.p, gpflow.probability_distributions.DiagonalGaussian
        ), "Must be DiagonalGaussian"

        # Psi 0
        if is_kernel(obj1) and feat1 is None and obj2 is None and feat2 is None:
            return tf.reduce_sum(self.quadrature(obj1.Kdiag))
        # Psi 1
        if is_kernel(obj1) and is_feature(feat1) and obj2 is None and feat2 is None:
            return self.quadrature(lambda x: obj1.K(x, feat1.Z))
        # Psi 2
        if is_kernel(obj1) and is_feature(feat1) and is_kernel(obj2) and is_feature(feat2):
            return self.quadrature(KxzKzx(obj1, feat1, obj2, feat2))

        print([type(x) for x in [obj1, feat1, obj2, feat2]])
        raise NotImplementedError()


class GaussHermiteExpectation(Expectation):
    @property
    def __name__(self):
        return f"gaussHermite({self.quadrature_points})"

    def __init__(self, quadrature_points, din=None):
        super().__init__()
        self.quadrature_points = quadrature_points
        self.p = None
        self.din = din

    def expectation(self, obj1, feat1, obj2, feat2):
        p = self.p
        if obj2 is None:
            eval_func = lambda x: gpflow.expectations.get_eval_func(obj1, feat1)(x)
        elif obj1 is None:
            raise NotImplementedError("First object cannot be None.")
        else:
            eval_func = lambda x: (
                    gpflow.expectations.get_eval_func(obj1, feat1, np.s_[:, :, None])(x)
                    * gpflow.expectations.get_eval_func(obj2, feat2, np.s_[:, None, :])(x)
            )
        if isinstance(p, gpflow.probability_distributions.DiagonalGaussian):
            if (
                    isinstance(obj1, gpflow.kernels.Kernel)
                    and isinstance(obj2, gpflow.kernels.Kernel)
                    and obj1.on_separate_dims(obj2)
            ):
                eKxz1 = self.expectation(obj1, feat1, None, None)
                eKxz2 = self.expectation(obj2, feat2, None, None)
                return eKxz1[:, :, None] * eKxz2[:, None, :]

            else:
                cov = tf.matrix_diag(p.cov)
        else:
            cov = p.cov
        return gpflow.quadrature.mvnquad(eval_func, p.mu, cov, self.quadrature_points, Din=self.din)


class UnscentedExpectation(QuadratureExpectation):
    @property
    def __name__(self):
        return f"unscented"

    def quadrature(self, f):
        n, d = tf.shape(self.p.mu)[0], tf.shape(self.p.mu)[1]
        n_sigma = tf.cast(2 * d, float_type)
        delta = tf.matrix_diag(tf.sqrt(tf.cast(d, float_type) * self.p.cov))

        f1 = f(self.p.mu[:, None, :] + delta)
        f2 = f(self.p.mu[:, None, :] - delta)

        return (tf.reduce_sum(f1, axis=1) + tf.reduce_sum(f2, axis=1)) / n_sigma


class MonteCarloExpectation(QuadratureExpectation):
    @property
    def __name__(self):
        return f"montecarlo({self.montecarlo_points})"

    def __init__(self, montecarlo_points):
        super().__init__()
        self.montecarlo_points = montecarlo_points
        self.samples = None
        self.p = None

    def load(self, p):
        n, d = tf.shape(p.mu)[0], tf.shape(p.mu)[1]
        samples = tf.random.normal((self.montecarlo_points, n, d), dtype=float_type)
        self.samples = tf.transpose(p.mu + tf.sqrt(p.cov) * samples, [1, 0, 2])
        self.p = p

    def quadrature(self, f):
        fsamples = f(self.samples)
        fxMean = tf.reduce_sum(fsamples, axis=1) / tf.constant(
            self.montecarlo_points, float_type
        )
        return fxMean
