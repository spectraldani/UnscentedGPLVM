import sys

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.decors import params_as_tensors

float_type = gpflow.settings.float_type
jitter_level = gpflow.settings.jitter


def tfprint(*args):
    print_op = tf.print(*args, output_stream=sys.stdout, summarize=-1)
    return tf.control_dependencies([print_op])


def logdet_chol(chol, name=None):
    return tf.multiply(
        tf.constant(2, dtype=float_type),
        tf.reduce_sum(tf.log(tf.linalg.diag_part(chol))),
        name=name,
    )


class GPLVM(gpflow.models.BayesianGPLVM):
    def __init__(
            self,
            expectation,
            X_mean,
            X_var,
            Y,
            kern,
            M,
            Z=None,
            X_prior_mean=None,
            X_prior_var=None,
            train_mode="GPLVM",
            predict_mode="GPLVM",
    ):
        gpflow.models.BayesianGPLVM.__init__(
            self, X_mean, X_var, Y, kern, M, Z, X_prior_mean, X_prior_var
        )
        self.expectation = expectation
        self.train_mode = train_mode
        self.predict_mode = predict_mode

    @params_as_tensors
    def _build_likelihood(self):
        if self.train_mode == "GPLVM":
            return self._GPLVM_build_likelihood()
        elif self.train_mode == "SGP":
            return self._SGP_build_likelihood()
        elif self.train_mode == "GP":
            return self._GP_build_likelihood()
        else:
            raise NotImplementedError("Unknown Train Mode " + self.train_mode)

    # Code adapted from GPFlow 1.4.1 original
    # https://github.com/GPflow/GPflow/blob/v1.4.1/gpflow/models/gplvm.py#L123-L166
    @params_as_tensors
    def _GPLVM_build_likelihood(self):
        pX = gpflow.probability_distributions.DiagonalGaussian(self.X_mean, self.X_var)

        num_inducing = len(self.feature)

        self.expectation.load(pX)
        psi0 = tf.reduce_sum(self.expectation.expectation(self.kern, None, None, None))
        psi1 = self.expectation.expectation(self.kern, self.feature, None, None)
        psi2 = tf.reduce_sum(
            self.expectation.expectation(self.kern, self.feature, self.kern, self.feature),
            axis=0,
        )

        Kuu = gpflow.features.Kuu(self.feature, self.kern, jitter=jitter_level)
        L = tf.cholesky(Kuu, name="L")
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)

        # B = tf.Print(B,[tf.linalg.tensor_diag_part(B)], summarize=1000)
        LB = tf.cholesky(
            B + jitter_level * tf.eye(num_inducing, dtype=float_type), name="LB"
        )
        log_det_B = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma

        # KL[q(x) || p(x)]
        dX_var = (
            self.X_var
            if len(self.X_var.get_shape()) == 2
            else tf.matrix_diag_part(self.X_var)
        )
        NQ = tf.cast(tf.size(self.X_mean), float_type)
        D = tf.cast(tf.shape(self.Y)[1], float_type)
        KL = (
                -0.5 * tf.reduce_sum(tf.log(dX_var))
                + 0.5 * tf.reduce_sum(tf.log(self.X_prior_var))
                - 0.5 * NQ
                + 0.5
                * tf.reduce_sum(
            (tf.square(self.X_mean - self.X_prior_mean) + dX_var) / self.X_prior_var
        )
        )

        # compute log marginal bound
        ND = tf.cast(tf.size(self.Y), float_type)
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += (
                -0.5
                * D
                * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.matrix_diag_part(AAT)))
        )
        bound -= KL
        return bound

    # Code adapted from GPFlow 1.4.1 original
    # https://github.com/GPflow/GPflow/blob/develop/gpflow/models/sgpr.py#L125-L160
    @params_as_tensors
    def _SGP_build_likelihood(self):
        num_inducing = len(self.feature)
        num_data = tf.cast(tf.shape(self.Y)[0], float_type)
        output_dim = tf.cast(tf.shape(self.Y)[1], float_type)

        err = self.Y - self.mean_function(self.X_mean)
        Kdiag = self.kern.Kdiag(self.X_mean)
        Kuf = gpflow.features.Kuf(self.feature, self.kern, self.X_mean)
        Kuu = gpflow.features.Kuu(self.feature, self.kern, jitter=jitter_level)
        L = tf.cholesky(Kuu, name="L")
        sigma = tf.sqrt(self.likelihood.variance)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        AAT = tf.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B, name="LB")
        Aerr = tf.matmul(A, err)
        c = tf.matrix_triangular_solve(LB, Aerr, lower=True) / sigma

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        bound += tf.negative(output_dim) * tf.reduce_sum(
            tf.log(tf.matrix_diag_part(LB))
        )
        bound -= 0.5 * num_data * output_dim * tf.log(self.likelihood.variance)
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * output_dim * tf.reduce_sum(Kdiag) / self.likelihood.variance
        bound += 0.5 * output_dim * tf.reduce_sum(tf.matrix_diag_part(AAT))

        return bound

    # Code adapted from GPFlow 1.4.1 original
    # https://github.com/GPflow/GPflow/blob/develop/gpflow/models/gpr.py#L64-L76
    @params_as_tensors
    def _GP_build_likelihood(self):
        K = (
                self.kern.K(self.X_mean)
                + tf.eye(tf.shape(self.X_mean)[0], dtype=float_type)
                * self.likelihood.variance
        )
        L = tf.cholesky(K)
        m = self.mean_function(self.X_mean)
        logpdf = gpflow.logdensities.multivariate_normal(self.Y, m, L)
        return tf.reduce_sum(logpdf)

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        if self.predict_mode == "GPLVM":
            return self._GPLVM_build_predict(Xnew)
        elif self.predict_mode == "GP":
            return self._GP_build_predict(Xnew)
        else:
            raise NotImplementedError("Unknown Predict Mode " + self.predict_mode)

    # Code adapted from GPFlow 1.4.1 original
    # https://github.com/GPflow/GPflow/blob/develop/gpflow/models/gpr.py#L80-L96
    def _GP_build_predict(self, Xnew, full_cov=False):
        y = self.Y - self.mean_function(self.X_mean)
        Kmn = self.kern.K(self.X_mean, Xnew)
        Kmm_sigma = (
                self.kern.K(self.X_mean)
                + tf.eye(tf.shape(self.X_mean)[0], dtype=float_type)
                * self.likelihood.variance
        )
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)
        f_mean, f_var = gpflow.conditionals.base_conditional(
            Kmn, Kmm_sigma, Knn, y, full_cov=full_cov, white=False
        )
        return f_mean + self.mean_function(Xnew), f_var

    # Code adapted from GPFlow 1.4.1 original
    # https://github.com/GPflow/GPflow/blob/develop/gpflow/models/gplvm.py#L169-L206
    def _GPLVM_build_predict(self, Xnew, full_cov=False):
        pX = gpflow.probability_distributions.DiagonalGaussian(self.X_mean, self.X_var)

        num_inducing = len(self.feature)
        psi1 = self.expectation(pX, (self.kern, self.feature))
        psi2 = tf.reduce_sum(
            self.expectation(pX, (self.kern, self.feature), (self.kern, self.feature)),
            axis=0,
        )
        Kuu = gpflow.features.Kuu(self.feature, self.kern, jitter=jitter_level)
        Kus = gpflow.features.Kuf(self.feature, self.kern, Xnew)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu, name="L")

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(B, name="LB")
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                    self.kern.K(Xnew)
                    + tf.matmul(tmp2, tmp2, transpose_a=True)
                    - tf.matmul(tmp1, tmp1, transpose_a=True)
            )
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = (
                    self.kern.Kdiag(Xnew)
                    + tf.reduce_sum(tf.square(tmp2), 0)
                    - tf.reduce_sum(tf.square(tmp1), 0)
            )
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var

    @params_as_tensors
    def _build_uncertain_predict(self, Xstarmu, Xstarvar):
        if self.predict_mode == "GPLVM":
            return self._GPVLM_build_uncertain_predict(Xstarmu, Xstarvar)
        elif self.predict_mode == "Girard":
            return self._Girard_build_uncertain_predict(Xstarmu, Xstarvar)
        else:
            raise NotImplementedError("Unknown Predict Mode")

    def _Girard_build_uncertain_predict(self, Xstarmu, Xstarvar):
        raise NotImplemented("Not yet implemented")

    # Code adapted from @javdrher fork of GPflow
    # https://github.com/javdrher/GPflow/blob/gplvm-predict-x/GPflow/gplvm.py#L216-L266
    def _GPVLM_build_uncertain_predict(self, Xstarmu, Xstarvar):
        pX = gpflow.probability_distributions.DiagonalGaussian(self.X_mean, self.X_var)
        pXstar = gpflow.probability_distributions.DiagonalGaussian(Xstarmu, Xstarvar)

        num_inducing = tf.shape(self.feature.Z)[0]  # M
        num_predict = tf.shape(Xstarmu)[0]  # N*
        num_out = self.output_dim  # p

        # Kernel expectations, w.r.t q(X) and q(X*)
        self.expectation.load(pX)
        psi1 = self.expectation.expectation(self.kern, self.feature, None, None)
        psi2 = tf.reduce_sum(
            self.expectation.expectation(self.kern, self.feature, self.kern, self.feature),
            axis=0,
        )
        #         psi1 = self.expectation(pX, (self.kern, self.feature))
        #         psi2 = tf.reduce_sum(
        #             self.expectation(pX, (self.kern, self.feature), (self.kern, self.feature)),
        #             axis=0,
        #         )

        self.expectation.load(pXstar)
        psi0star = tf.reduce_sum(self.expectation.expectation(self.kern, None, None, None))
        psi1star = self.expectation.expectation(self.kern, self.feature, None, None)
        psi2star = self.expectation.expectation(self.kern, self.feature, self.kern, self.feature)
        #         psi0star = tf.reduce_sum(self.expectation(pXstar, self.kern))
        #         psi1star = self.expectation(pXstar, (self.kern, self.feature))
        #         psi2star = self.expectation(
        #             pXstar, (self.kern, self.feature), (self.kern, self.feature)
        #         )

        Kuu = (
                self.kern.K(self.feature.Z)
                + tf.eye(num_inducing, dtype=float_type) * jitter_level
        )  # M x M
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu, name="L")  # M x M

        A = (
                tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        )  # M x N
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)  # M x M
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=float_type)
        LB = tf.cholesky(
            B + tf.eye(num_inducing, dtype=float_type) * jitter_level, name="LB"
        )  # M x M
        c = (
                tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma
        )  # M x p
        tmp1 = tf.matrix_triangular_solve(L, tf.transpose(psi1star), lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)

        # All of these: N* x M x M
        L3 = tf.tile(tf.expand_dims(L, 0), [num_predict, 1, 1])
        LB3 = tf.tile(tf.expand_dims(LB, 0), [num_predict, 1, 1])
        tmp3 = tf.matrix_triangular_solve(
            LB3, tf.matrix_triangular_solve(L3, tf.expand_dims(psi1star, -1))
        )
        tmp4 = tf.matmul(tmp3, tmp3, transpose_b=True)
        tmp5 = tf.matrix_triangular_solve(
            L3, tf.transpose(tf.matrix_triangular_solve(L3, psi2star), perm=[0, 2, 1])
        )
        tmp6 = tf.matrix_triangular_solve(
            LB3, tf.transpose(tf.matrix_triangular_solve(LB3, tmp5), perm=[0, 2, 1])
        )

        c3 = tf.tile(tf.expand_dims(c, 0), [num_predict, 1, 1])  # N* x M x p
        TT = tf.trace(tmp5 - tmp6)  # N*
        diagonals = tf.einsum(
            "ij,k->ijk", tf.eye(num_out, dtype=float_type), psi0star - TT
        )  # p x p x N*
        covar1 = tf.matmul(
            c3, tf.matmul(tmp6 - tmp4, c3), transpose_a=True
        )  # N* x p x p
        covar2 = tf.transpose(diagonals, perm=[2, 0, 1])  # N* x p x p
        covar = covar1 + covar2
        return mean + self.mean_function(Xstarmu), covar

    @gpflow.decors.autoflow((float_type, [None, None]), (float_type, [None, None]))
    def predict_y_uncertain(self, Xstarmu, Xstarvar):
        mean, covar = self._build_uncertain_predict(Xstarmu, Xstarvar)
        return self.likelihood.predict_mean_and_var(mean, covar)
