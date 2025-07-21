import torch
import numpy as np
import scipy.sparse as sp
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def sample_mask(shape, p, p_noise, min_seq, max_seq, rng, pattern="point"):
    """
    Simplified placeholder for the sample_mask function.
    Creates a boolean mask where True indicates a value to be masked (made missing).
    """
    # print(f"sample_mask called with shape: {shape}, p: {p}, p_noise: {p_noise}, pattern: {pattern}")
    if pattern == "block":
        # Simplified block missing: choose some sequences to mask entirely
        # This is a very naive block implementation
        mask = np.zeros(shape, dtype=bool)
        num_nodes, num_timesteps, num_features = shape
        for node_idx in range(num_nodes):
            for feat_idx in range(num_features):
                if rng.random() < p: # Probability to start a block
                    block_len = rng.integers(min_seq, max_seq + 1)
                    start = rng.integers(0, num_timesteps - block_len + 1)
                    mask[node_idx, start:start+block_len, feat_idx] = True
        # Add some random noise on top if p_noise is high
        noise_mask = rng.random(shape) < p_noise
        return np.logical_or(mask, noise_mask)
    else: # Point missing
        return rng.random(shape) < p_noise
def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()
def load_data(data_name, ratio = [0.7, 0.1]):
    if data_name == 'US':
        adj_mx = np.load('udata/adj_mx.npy')
        od_power = np.load('udata/od_pair.npy')
        od_power = od_power/(1.5*od_power.max())
        od_power[od_power < 0.1] = 0
        for i in range(70):
            od_power[i, i] = 1
        adj = [asym_adj(adj_mx), asym_adj(od_power), asym_adj(od_power.T)]
        data = np.load('udata/udelay.npy')
        wdata = np.load('udata/weather2016_2021.npy')  
    if data_name == 'China':
        adj_mx = np.load('cdata/dist_mx.npy')
        od_power = np.load('cdata/od_mx.npy')
        od_power = od_power/(1.5*od_power.max())
        od_power[od_power < 0.1] = 0
        for i in range(50):
            od_power[i, i] = 1
        adj = [asym_adj(adj_mx), asym_adj(od_power), asym_adj(od_power.T)]
        data = np.load('cdata/delay.npy')
        data[data<-15] = -15
        wdata = np.load('cdata/weather_cn.npy')
    data = np.clip(data, -30, 30)
    training_data = data[:, :int(ratio[0]*data.shape[1]) ,:]
    val_data = data[:,int(ratio[0]*data.shape[1]):int((ratio[0] + ratio[1])*data.shape[1]),:]
    test_data = data[:, int((ratio[0] + ratio[1])*data.shape[1]):, :]
    training_w = wdata[:, :int(ratio[0]*data.shape[1])]
    val_w = wdata[:,int(ratio[0]*data.shape[1]):int((ratio[0] + ratio[1])*data.shape[1])]
    test_w = wdata[:, int((ratio[0] + ratio[1])*data.shape[1]):]    
    return adj, training_data, val_data, test_data, training_w, val_w, test_w
def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_wmae(preds, labels, weights, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask * weights
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def get_lap_pos_enc(A,d_model):
    '''

    :param adj_mx:  (N,N)
    :num_of_vertices: N
    :return:
        '''
    num_nodes=A.shape[0]
    N = np.diag(1.0/np.sum(A, axis=1))
#     L = np.dot(np.dot(N, A),N)
    L=sp.eye(num_nodes)-np.dot(np.dot(N, A),N)
    EigVal, EigVec = np.linalg.eig(L)
    idx = EigVal.argsort()
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    lap_pos_enc=EigVec[:,1:d_model+1]
    return lap_pos_enc 
def test_error(y_predict, y_test):
    """
    Calculates MAE, RMSE, R2.
    :param y_test:
    :param y_predict.
    :return:
    """
    err = y_predict - y_test
    MAE = np.mean(np.abs(err[~np.isnan(err)]))
    
    s_err = err**2
    RMSE = np.sqrt(np.mean((s_err[~np.isnan(s_err)])))
    
    test_mean = np.mean((y_test[~np.isnan(y_test)]))
    m_err = (y_test - test_mean)**2
    R2 = 1 - np.sum(s_err[~np.isnan(s_err)])/np.sum(m_err[~np.isnan(m_err)])
    
    return MAE, RMSE, R2