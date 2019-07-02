import gpflow
import tensorflow as tf
import numpy as np

class MLP(gpflow.kernels.Kernel):
    def __init__(self, input_dim, layers_width, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)

        pipeline = [input_dim] + layers_width
        self.layers = pipeline
        pipeline = list(zip(pipeline, pipeline[1:]))

        self.W = gpflow.params.ParamList([
            np.random.normal(size=shape, scale=np.sqrt(2/shape[0])) for shape in pipeline
        ], name='Weights')
        self.b = gpflow.params.Parameter(1e-2*np.ones(len(pipeline)), name='Bias')
        self.rbf = gpflow.kernels.SquaredExponential(layers_width[-1], ARD=True)

    def clone(self):
        c = MLP(self.layers[0], self.layers[1:], self.active_dims)
        for i in range(len(self.W)):
            c.W[i].assign(self.W[i].value)
        c.b.assign(self.b.value)
        return c

    @gpflow.decors.params_as_tensors
    def _transform(self, X):
        Y = X
        for i in range(0,len(self.W)-1):
            Y = tf.nn.relu(tf.matmul(Y,self.W[i]) + self.b[i])
        #Y = tf.matmul(Y,self.W[-1]) + self.b[-1]
        Y = tf.nn.tanh(tf.matmul(Y,self.W[-1]) + self.b[-1])
        return Y

    @gpflow.decors.params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        lX = self._transform(X)
        lX2 = self._transform(X2)
        return self.rbf.K(lX,lX2)

    @gpflow.decors.params_as_tensors
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)

        lX = self._transform(X)
        return self.rbf.Kdiag(lX)


class SpectralMixture(gpflow.kernels.Kernel):
    two_pi = tf.constant(2*np.pi, dtype=gpflow.settings.float_type)
    minus_two_pi_squared = tf.constant(-2*np.pi**2, dtype=gpflow.settings.float_type)

    def get_initial_parameters(X,y,num_mixtures):
        input_dim = X.shape[1]

        sorted_by_dimension = np.sort(X,axis=0)
        largest_interval = sorted_by_dimension[-1]-sorted_by_dimension[0]
        smallest_interval = np.abs(sorted_by_dimension[1:]-sorted_by_dimension[:-1])
        smallest_interval[smallest_interval == 0] = np.inf
        smallest_interval = np.amin(smallest_interval, axis=0)


        means = np.random.rand(num_mixtures,input_dim) * (0.5/smallest_interval)

        lengthscales = np.abs(largest_interval * np.random.randn(num_mixtures,input_dim))
        covariance = 1/lengthscales

        weights = np.std(y,axis=0)/num_mixtures * np.ones(num_mixtures)

        return weights, means, covariance

    def __init__(self, X, y, num_mixtures=1, active_dims=None, name=None):
        super().__init__(X.shape[1], active_dims, name=name)

        weights, means, covariance = SpectralMixture.get_initial_parameters(X,y,num_mixtures)
        self.weights = gpflow.params.Parameter(weights, transform=gpflow.transforms.positive)
        self.means = gpflow.params.Parameter(means, transform=gpflow.transforms.positive)
        self.covariance = gpflow.params.Parameter(covariance, transform=gpflow.transforms.positive)

    @gpflow.decors.params_as_tensors
    def _tau(self, X, X2):
        if X2 is None:
            X2 = X
        return X[:,tf.newaxis,:]-X2[tf.newaxis,:,:]


    @gpflow.decors.params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        tau = self._tau(X,X2)

        exp_term = tf.reduce_prod(
            tf.exp(self.minus_two_pi_squared * tau**2 * self.covariance[:,tf.newaxis,tf.newaxis,:]), axis=3
        )

#         tf.einsum('mi,abd->mab',self.means,tau)
        cos_term = tf.cos(self.two_pi * tf.tensordot(self.means,tau,[[1],[2]]))
        return tf.reduce_sum(self.weights[:,tf.newaxis,tf.newaxis] * cos_term * exp_term, axis=0)

    @gpflow.decors.params_as_tensors
    def Kdiag(self, X):
        return tf.fill(tf.stack([tf.shape(X)[0]]),tf.reduce_sum(self.weights,0))

