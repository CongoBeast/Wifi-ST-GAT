import torch
import numpy as np


def scaled_laplacian(A):
    n = A.shape[0]
    d = np.sum(A, axis=1)
    L = np.diag(d) - A
    for i in range(n):
        for j in range(n):
            if d[i] > 0 and d[j] > 0:
                L[i, j] /= np.sqrt(d[i] * d[j])
    lam = np.linalg.eigvals(L).max().real
    return 2 * L / lam - np.eye(n)


def cheb_poly(L, Ks):
    n = L.shape[0]
    LL = [np.eye(n), L[:]]
    for i in range(2, Ks):
        LL.append(np.matmul(2 * L, LL[-1]) - LL[-2])
    return np.asarray(LL)


def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, MAPE, RMSE

      
def MAPE(v, v_):
    '''
    Mean absolute percentage error.
    :param v: torch.Tensor or float, ground truth.
    :param v_: torch.Tensor or float, prediction.
    :return: float, MAPE averaged over all elements of input.
    '''
    return torch.mean(torch.abs(v_ - v) / (v + 1e-5))

def MAE(v, v_):
    '''
    Mean absolute error.
    :param v: torch.Tensor or float, ground truth.
    :param v_: torch.Tensor or float, prediction.
    :return: float, MAE averaged over all elements of input.
    '''
    return torch.mean(torch.abs(v_ - v))

def RMSE(v, v_):
    '''
    Root mean squared error.
    :param v: torch.Tensor or float, ground truth.
    :param v_: torch.Tensor or float, prediction.
    :return: float, RMSE averaged over all elements of input.
    '''
    return torch.sqrt(torch.mean((v_ - v) ** 2))

def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    :param x: np.ndarray, input to be recovered.
    :param mean: float, the value of mean.
    :param std: float, the value of standard deviation.
    :return: np.ndarray, z-score inverse array.
    '''
    return x * std + mean
