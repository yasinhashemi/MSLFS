import tensorly as tl 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")
tl.set_backend('pytorch')
plt.rcParams["figure.dpi"] = 150  # For display
plt.rcParams["savefig.dpi"] = 300  # For saved figures

# Visualization of Selected Features
def feature_visualization(data, top_ind_1, top_ind_2):
    plt.figure(figsize= (6, 8))
    plt.imshow(data, cmap= 'plasma')
    plt.xticks([])
    plt.yticks([])
    plt.scatter(top_ind_1[:, None].repeat(1, top_ind_2.numel()).cpu(), top_ind_2[None, :].repeat(top_ind_1.numel(), 1).cpu(), marker='x', color='green', s= 80, linewidths= 5)

# t-SNE Plots
def t_sne(T, Y):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_embedded = tsne.fit_transform(T.cpu())
    plt.figure(figsize= (8, 6))
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c= Y, cmap= 'tab10', s= 60)
    plt.show()