import numpy as np


def diagonalize_predict_var(predict_var):
    if predict_var.shape[1] == predict_var.shape[0]:
        return np.diag(predict_var).reshape(-1, 1)
    else:
        return predict_var


def root_mean_squared_error(observed_value, predict_mean, predict_var=None):
    se = (observed_value - predict_mean) ** 2
    return np.sqrt(np.mean(se, axis=0))


def negative_log_predictive_density(observed_value, predict_mean, predict_var):
    n = observed_value.shape[0]
    predict_var = diagonalize_predict_var(predict_var)
    inner = np.log(predict_var) + (observed_value - predict_mean) ** 2 / predict_var
    return 0.5 * (np.log(2 * np.pi) + np.mean(inner, axis=0))


def mean_relative_absolute_error(observed_value, predicted_mean, predict_var=None):
    ae = np.abs(observed_value - predicted_mean)
    return np.mean(ae / np.abs(observed_value), axis=0)
