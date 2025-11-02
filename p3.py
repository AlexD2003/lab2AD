import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from sklearn.metrics import balanced_accuracy_score

def run_ex3():
    centers = [(-10, -10), (10, 10)]
    X1, y1 = make_blobs(n_samples=200, n_features=2, centers=[centers[0]], cluster_std=2.0, random_state=0)
    X2, y2 = make_blobs(n_samples=100, n_features=2, centers=[centers[1]], cluster_std=6.0, random_state=1)
    X = np.vstack([X1, X2])
    #ground truth: treat none as anomaly
    contamination = 0.07

    for k in [5, 20, 50]:
        clf_knn = KNN(n_neighbors=k)
        clf_knn.fit(X)
        labels_knn = clf_knn.labels_

        clf_lof = LOF(n_neighbors=k, contamination=contamination)
        clf_lof.fit(X)
        labels_lof = clf_lof.labels_

        #plot
        fig, axs = plt.subplots(1,2, figsize=(10,4))
        axs[0].scatter(X[:,0], X[:,1], c=labels_knn, cmap='coolwarm', s=40, edgecolor='k')
        axs[0].set_title(f'KNN n_neighbors={k}')
        axs[1].scatter(X[:,0], X[:,1], c=labels_lof, cmap='coolwarm', s=40, edgecolor='k')
        axs[1].set_title(f'LOF n_neighbors={k}')
        plt.suptitle('KNN vs LOF on clusters with different densities')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    run_ex3()
