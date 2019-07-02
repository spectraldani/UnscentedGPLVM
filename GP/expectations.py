import gpflow
import tensorflow as tf
import numpy as np
settings = gpflow.settings
tfd = tf.contrib.distributions

def wrapExpectation(f):
    def e(p, obj1, obj2=None):
        if isinstance(p, tuple):
            assert len(p) == 2
            if p[1].shape.ndims == 2:
                p = gpflow.probability_distributions.DiagonalGaussian(*p)
            elif p[1].shape.ndims == 3:
                p = gpflow.probability_distributions.Gaussian(*p)
            elif p[1].shape.ndims == 4:
                p = gpflow.probability_distributions.MarkovGaussian(*p)

        if isinstance(obj1, tuple):
            obj1, feat1 = obj1
        else:
            feat1 = None

        if isinstance(obj2, tuple):
            obj2, feat2 = obj2
        else:
            feat2 = None
        return f(p, obj1, feat1, obj2, feat2)
    e.__name__ = f.__name__
    return e


def unscentedTransformation(f, p, covariance=False):
    n,d = tf.shape(p.mu)[0], tf.shape(p.mu)[1]
    nSigmas = 2*d
    sqrtCov = tf.matrix_diag(tf.sqrt(tf.cast(d, settings.float_type) * p.cov))
    delta = tf.concat([sqrtCov,tf.negative(sqrtCov)], axis=1)
    sigmas = tf.reshape(tf.tile(p.mu, [1,nSigmas]),[n,nSigmas,d]) + delta
    fsig = tf.map_fn(f, sigmas)
    fxMean = tf.reduce_sum(fsig, 1)/tf.cast(nSigmas, settings.float_type)
#     print(sigmas.shape, fsig.shape, fxMean.shape)
    return fxMean

def montecarlo(montecarlo_points):
    def e(f, p):
        n,d = tf.shape(p.mu)[0], tf.shape(p.mu)[1]
        samples = tf.random.normal((montecarlo_points, n, d), dtype=settings.float_type)
        samples = p.mu + tf.sqrt(p.cov) * samples
        samples = tf.transpose(samples,[1,0,2])
        fsamples = tf.map_fn(f, samples)
        fxMean = tf.reduce_sum(fsamples, 1)/tf.constant(montecarlo_points, settings.float_type)
#         print(samples.shape, fsamples.shape, fxMean.shape)
        return fxMean
    e.__name__ = f'montecarlo({montecarlo_points})'
    return e

@wrapExpectation
def analyticExpectation(p, obj1, feat1, obj2, feat2):
    return gpflow.expectations._expectation(p, obj1, feat1, obj2, feat2, nghp=0)
analyticExpectation.__name__ = 'analytic'

def gaussHermiteExpectation(quadrature_points, Din=None):
    def e(p, obj1, obj2=None):
        if isinstance(p, tuple):
            assert len(p) == 2
            if p[1].shape.ndims == 2:
                p = gpflow.probability_distributions.DiagonalGaussian(*p)
            elif p[1].shape.ndims == 3:
                p = gpflow.probability_distributions.Gaussian(*p)
            elif p[1].shape.ndims == 4:
                p = gpflow.probability_distributions.MarkovGaussian(*p)
        if isinstance(obj1, tuple):
            obj1, feat1 = obj1
        else:
            feat1 = None
        if isinstance(obj2, tuple):
            obj2, feat2 = obj2
        else:
            feat2 = None
        if obj2 is None:
            eval_func = lambda x: gpflow.expectations.get_eval_func(obj1, feat1)(x)
        elif obj1 is None:
            raise NotImplementedError("First object cannot be None.")
        else:
            eval_func = lambda x: (gpflow.expectations.get_eval_func(obj1, feat1, np.s_[:, :, None])(x) *
                                   gpflow.expectations.get_eval_func(obj2, feat2, np.s_[:, None, :])(x))
        if isinstance(p, gpflow.probability_distributions.DiagonalGaussian):
            if isinstance(obj1, gpflow.kernels.Kernel) and isinstance(obj2, gpflow.kernels.Kernel) and obj1.on_separate_dims(obj2):
                eKxz1 = e(p, (obj1, feat1))
                eKxz2 = e(p, (obj2, feat2))
                return eKxz1[:, :, None] * eKxz2[:, None, :]

            else:
                cov = tf.matrix_diag(p.cov)
        else:
            cov = p.cov
        return gpflow.quadrature.mvnquad(eval_func, p.mu, cov, quadrature_points, Din=Din)

    e.__name__ = f'gaussHermite({quadrature_points})'
    return e

def KxzKzx(k1,feat1, k2,feat2):
    num_inducing = len(feat1)
    def inner(x):
        K1xz = k1.K(x, feat1.Z)
        K2xz = k2.K(x, feat2.Z)
        Kxz2 = tf.matmul(K1xz,K2xz,transpose_a=True)
        return tf.reshape(Kxz2, [1,len(feat1),len(feat2)])
    return inner
def isFeature(o):
    return isinstance(o,gpflow.features.InducingFeature)
def isKernel(o):
    return isinstance(o,gpflow.kernels.Kernel)

def wrapQuadrature(f):
    @wrapExpectation
    def e(p, obj1, feat1, obj2, feat2):
        assert isinstance(p, gpflow.probability_distributions.DiagonalGaussian), 'Must be DiagonalGaussian'
        cov = p.cov
        mu = p.mu

        # Psi 0
        if (isKernel(obj1) and feat1 is None and obj2 is None and feat2 is None):
            return tf.reduce_sum(f(obj1.Kdiag, p))
        # Psi 1
        if (isKernel(obj1) and isFeature(feat1) and obj2 is None and feat2 is None):
            return f(lambda x: obj1.K(x, feat1.Z), p)
        # Psi 2
        if (isKernel(obj1) and isFeature(feat1) and isKernel(obj2) and isFeature(feat2)):
            return f(KxzKzx(obj1, feat1, obj2, feat2), p)

        print([type(x) for x in [obj1,feat1,obj2,feat2]])
        raise NotImplementedError()
    e.__name__ = f.__name__
    return e

unscentedExpectation = wrapQuadrature(unscentedTransformation)
unscentedExpectation.__name__ = 'unscented'

def montecarloExpectation(points):
    return wrapQuadrature(montecarlo(points))
