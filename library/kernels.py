import gpflow
import tensorflow as tf
import numpy as np


class MLP(gpflow.kernels.Kernel):
    def __init__(self, input_dim, layers_width, active_dims=None, name=None):
        super().__init__(input_dim, active_dims, name=name)

        pipeline = [input_dim] + layers_width
        self.layers = pipeline
        pipeline = list(zip(pipeline, pipeline[1:]))

        self.W = gpflow.params.ParamList(
            [
                np.random.normal(size=shape, scale=np.sqrt(2 / shape[0]))
                for shape in pipeline
            ],
            name="Weights",
        )

        self.b = gpflow.params.Parameter(1e-2 * np.ones(len(pipeline)), name="Bias")
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
        for i in range(0, len(self.W) - 1):
            Y = tf.nn.relu(tf.matmul(Y, self.W[i]) + self.b[i])
        # Y = tf.matmul(Y,self.W[-1]) + self.b[-1]
        Y = tf.nn.tanh(tf.matmul(Y, self.W[-1]) + self.b[-1])
        return Y

    @gpflow.decors.params_as_tensors
    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        lX = self._transform(X)
        lX2 = self._transform(X2)
        return self.rbf.K(lX, lX2)

    @gpflow.decors.params_as_tensors
    def Kdiag(self, X, presliced=False):
        if not presliced:
            X, _ = self._slice(X, None)

        lX = self._transform(X)
        return self.rbf.Kdiag(lX)
