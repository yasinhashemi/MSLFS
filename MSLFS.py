import torch 
import tensorly as tl 
import scipy
import numpy as np
from tensorly.tenalg import mode_dot
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from torch.linalg import norm
from utils import preprocessing

class TFS(preprocessing):
    def __init__(self, dataset_name, fs1, fs2, alpha, beta, gamma,  k_neigh= 5, sigma= 1000, MaxIter= 5, for_convergence= False):
        super().__init__(dataset_name, k_neigh, sigma)
        self.I_1, self.I_2, self.I_3 = list(map(lambda x: int(x), self.X.shape)) # Tensor Data Dimensions
        self.A_1, self.B_1, self.L_1 = self.knn_graph(mode= 1) # Weight, Degree, and Laplacian Matrices Associated to Mode-1
        self.A_2, self.B_2, self.L_2 = self.knn_graph(mode= 2) # Weight, Degree, and Laplacian Matrices Associated to Mode-2
        self.X_1, self.X_2 = self.X_1.to('cuda'), self.X_2.to('cuda') # Mode-1 and Mode-2 Unfoldings of Tensor data
        self.fs1, self.fs2 = fs1, fs2 # Number of Selected Slices along Mode-1 and Mode-2, Respectively
        # Initialization
        self.W_1 = torch.rand((self.fs1, self.I_1), device= 'cuda')
        self.W_2 = torch.rand((self.fs2, self.I_2), device= 'cuda')
        self.H_1 = torch.rand((self.I_1, self.fs1), device= 'cuda')
        self.H_2 = torch.rand((self.I_2, self.fs2), device= 'cuda')
        self.MaxIter, self.alpha, self.beta, self.gamma = MaxIter, alpha, beta, gamma # Hyperparameters
        self.dataset_name = dataset_name
        self.for_convergence = for_convergence
        self.convergence_arr = torch.zeros(self.MaxIter, dtype= torch.float32)

    # Update U_1
    def update_U_1(self):
        t1 = torch.norm(self.W_1, dim= 0)
        self.U_1 = torch.diag(torch.div(1, torch.maximum(t1, torch.full_like(t1, 1e-8)))).cuda()

    # Update U_2
    def update_U_2(self):
        t2 = torch.norm(self.W_2, dim= 0)
        self.U_2 = torch.diag(torch.div(1, torch.maximum(t2, torch.full_like(t2, 1e-8)))).cuda()

    # Update W_1
    def update_W_1(self):
        z = self.W_2 @ self.W_2.T
        self.P_1 = torch.block_diag(*[self.H_2 @ self.W_2 for _ in range(self.I_3)]).to_sparse_coo().coalesce().to('cuda')
        top = torch.sparse.mm(self.H_1.T @ self.X_1, self.P_1) @ self.X_1.T + self.gamma * torch.trace(z) * self.W_1
        bottom = torch.sparse.mm(self.H_1.T @ self.H_1 @ self.W_1 @ self.X_1, self.P_1.T) @ torch.sparse.mm(self.P_1, self.X_1.T) + self.beta * torch.trace(self.W_2 @ self.U_2 @ self.W_2.T) * self.W_1 @ self.U_1 \
        + self.gamma * torch.trace(z @ z) * self.W_1 @ self.W_1.T @ self.W_1
        self.W_1 = self.W_1 * (top / torch.maximum(bottom, torch.full_like(bottom, 1e-10)))

    # Update H_1
    def update_H_1(self):
        top = torch.sparse.mm(self.X_1, self.P_1) @ self.X_1.T @ self.W_1.T + self.alpha * torch.trace(self.H_2.T @ self.L_2 @ self.H_2) * self.A_1 @ self.H_1
        bottom = torch.sparse.mm(self.H_1 @ self.W_1 @ self.X_1, self.P_1.T) @ torch.sparse.mm(self.P_1, self.X_1.T @ self.W_1.T) + self.alpha * torch.trace(self.H_2.T @ self.L_2 @ self.H_2) * self.B_1 @ self.H_1
        self.H_1 = self.H_1 * (top / torch.maximum(bottom, torch.full_like(bottom, 1e-10)))

    # Update W_2
    def update_W_2(self):
        p = self.W_1 @ self.W_1.T
        self.P_2 = torch.block_diag(*[self.H_1 @ self.W_1 for _ in range(self.I_3)]).to_sparse_coo().coalesce().to('cuda')
        top = torch.sparse.mm(self.H_2.T @ self.X_2, self.P_2) @ self.X_2.T + self.gamma * torch.trace(p) * self.W_2
        bottom = torch.sparse.mm(self.H_2.T @ self.H_2 @ self.W_2 @ self.X_2, self.P_2.T) @ torch.sparse.mm(self.P_2, self.X_2.T) + self.beta * torch.trace(self.W_1 @ self.U_1 @ self.W_1.T) * self.W_2 @ self.U_2 \
        + self.gamma * torch.trace(p @ p) * self.W_2 @ self.W_2.T @ self.W_2
        self.W_2 = self.W_2 * (top / torch.maximum(bottom, torch.full_like(bottom, 1e-10)))

    # Update H_2
    def update_H_2(self):
        top = torch.sparse.mm(self.X_2, self.P_2) @ self.X_2.T @ self.W_2.T + self.alpha * torch.trace(self.H_1.T @ self.L_1 @ self.H_1) * self.A_2 @ self.H_2
        bottom = torch.sparse.mm(self.H_2 @ self.W_2 @ self.X_2, self.P_2.T) @ torch.sparse.mm(self.P_2, self.X_2.T @ self.W_2.T) + self.alpha * torch.trace(self.H_1.T @ self.L_1 @ self.H_1) * self.B_2 @ self.H_2
        self.H_2 = self.H_2 * (top / torch.maximum(bottom, torch.full_like(bottom, 1e-10)))

    # Objective function
    def objective(self):
        return 0.5 * norm(torch.flatten(self.X.cuda() - mode_dot(mode_dot(self.X.cuda(), self.H_1 @ self.W_1, mode= 0), self.H_2 @ self.W_2, 1))) ** 2 \
        + self.alpha / 2 * torch.trace(self.H_2.T @ self.L_2 @ self.H_2) * torch.trace(self.H_1.T @ self.L_1 @ self.H_1) + self.beta / 2 * torch.trace(self.W_2 @ self.U_2 @ self.W_2.T) * torch.trace(self.W_1 @ self.U_1 @ self.W_1.T)

    # Training
    def training(self):
        for i in range(self.MaxIter):
            self.update_U_1()
            self.update_U_2()
            self.update_W_1()
            self.update_H_1()
            self.update_W_2()
            self.update_H_2()
            
    # Feature selection
    def feature_selection(self):
        score_1 = norm(self.W_1, dim= 0)
        score_2 = norm(self.W_2, dim= 0)
        _, top_ind_1 = torch.topk(score_1, self.fs1)
        _, top_ind_2 = torch.topk(score_2, self.fs2)
        # Selecting Intersection Mode-3 Fibers
        self.T = self.X.cuda()[top_ind_1[:, None], top_ind_2[None, :]].permute(2, 0, 1).view(self.I_3, -1)

    # Clustering
    def clustering(self):
        clusters = KMeans(n_clusters= self.labels_num).fit(self.T.cpu())
        return clusters.labels_
    
    # Evaluation
    def clustering_accuracy(self):
        real_labels = np.asarray(self.Y.numpy())
        labels = np.asarray(self.clustering())
        permutation = []
        n_clusters = len(np.unique(real_labels))

        labels = np.unique(labels, return_inverse=True)[1]
        for i in range(n_clusters):
            idx = labels == i
            if np.sum(idx) != 0:
                new_label = scipy.stats.mode(real_labels[idx], keepdims= True)[0][0]
                permutation.append(new_label)
        new_labels = [permutation[label] for label in labels]
        return accuracy_score(real_labels, new_labels), normalized_mutual_info_score(real_labels, new_labels), adjusted_rand_score(real_labels, new_labels)
