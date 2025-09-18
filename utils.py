import torch 
import tensorly as tl 
import faiss
import numpy as np
from scipy.io import loadmat
tl.set_backend('pytorch')

class preprocessing:
    def __init__(self, dataset_name, k_neigh, sigma):
        data = loadmat(f"C:\\Users\\Kavoosh\\Desktop\\Reshaped_Data\\Reshaped_Datasets\\{dataset_name}.mat")
        self.X = torch.tensor(data['X'], dtype= torch.float32)

        if self.X.max() > 1:
            self.X /= self.X.max()

    
        self.X_1 = tl.unfold(self.X, mode= 0)
        self.X_2 = tl.unfold(self.X, mode= 1)
        self.X_3 = tl.unfold(self.X, mode= 2)

        self.Y = torch.tensor(data['Y']).ravel() - 1 if data['Y'].min() > 0 else torch.tensor(data['Y']).ravel()
        self.labels_num = self.Y.unique().numel()
        self.k_neigh = k_neigh
        self.sigma = sigma

    def knn_graph(self, mode):
        if mode == 1:
            X_ = self.X_1
        else:
            X_ = self.X_2

        n, d = X_.shape
        index = faiss.IndexFlatL2(d)
        index.add(X_)

        distances, indices = index.search(X_, self.k_neigh + 1)
        distances, indices = np.exp(-distances[:, 1:] / self.sigma), indices[:, 1:]

        W = np.zeros((n, n))
        W[np.arange(n)[:, np.newaxis], indices] = distances
        W = np.add(W, W.T) / 2
        D = np.diag(W.sum(axis= 1))
        L = D - W

        return torch.tensor(W, dtype= torch.float32, device= 'cuda'), torch.tensor(D, dtype= torch.float32, device= 'cuda'), torch.tensor(L, dtype= torch.float32, device= 'cuda')
