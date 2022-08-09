import numpy as np


def MAPE(v, v_):
    mape = (np.abs(v_ - v) / np.abs(v) + 1e-5).astype(np.float64)
    mape = np.where(mape > 5, 5, mape)
    return np.mean(mape)


def MAE(v, v_):
    return np.mean(np.abs(v_ - v)).astype(np.float64)


def RMSE(v, v_):
    return np.sqrt(np.mean((v_ - v) ** 2)).astype(np.float64)


def evaluate(y, y_hat):
    return MAPE(y, y_hat), MAE(y, y_hat), RMSE(y, y_hat)
