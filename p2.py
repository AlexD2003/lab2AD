import numpy as np
import matplotlib.pyplot as plt
from pyod.utils import data as pyod_data
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score

def run_ex2(n_train=400, n_test=200, contamination=0.1, n_clusters=2):
    #generate data
    try:
        X_train, y_train, X_test, y_test = pyod_data.generate_data(n_train=n_train, n_test=n_test,
                                                                   n_features=2, contamination=contamination,
                                                                   n_clusters=n_clusters, cluster_std=0.6,
                                                                   random_state=42, return_clusters=False)
    except Exception as e:
        from sklearn.datasets import make_blobs
        X_full, y_full = make_blobs(n_samples=n_train+n_test, centers=n_clusters, n_features=2, cluster_std=1.0, random_state=42)
        #mark 10% random as outliers
        n_out = int((n_train+n_test) * contamination)
        out_idx = np.random.choice(np.arange(len(X_full)), size=n_out, replace=False)
        y_full = np.zeros(len(X_full), dtype=int)
        y_full[out_idx] = 1
        X_train, X_test = X_full[:n_train], X_full[n_train:]
        y_train, y_test = y_full[:n_train], y_full[n_train:]

    neighbors_list = [5, 10, 20, 50]
    results = {}
    for k in neighbors_list:
        clf = KNN(n_neighbors=k, method='largest')
        clf.fit(X_train)
        y_train_pred = clf.labels_            #0 inliers, 1 outliers
        y_test_pred = clf.predict(X_test)     #returns labels for test

        ba_train = balanced_accuracy_score(y_train, y_train_pred)
        ba_test = balanced_accuracy_score(y_test, y_test_pred)
        results[k] = {'ba_train': ba_train, 'ba_test': ba_test,
                      'y_train_pred': y_train_pred, 'y_test_pred': y_test_pred}

        #plot the 4 subplots for this k
        fig, axs = plt.subplots(1,4, figsize=(20,4))
        axs[0].scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='coolwarm', s=30, edgecolor='k')
        axs[0].set_title('GT train')
        axs[1].scatter(X_train[:,0], X_train[:,1], c=y_train_pred, cmap='coolwarm', s=30, edgecolor='k')
        axs[1].set_title(f'Pred train (k={k}) BA={ba_train:.3f}')
        axs[2].scatter(X_test[:,0], X_test[:,1], c=y_test, cmap='coolwarm', s=30, edgecolor='k')
        axs[2].set_title('GT test')
        axs[3].scatter(X_test[:,0], X_test[:,1], c=y_test_pred, cmap='coolwarm', s=30, edgecolor='k')
        axs[3].set_title(f'Pred test (k={k}) BA={ba_test:.3f}')
        plt.tight_layout()
        plt.show()

    #print summary 
    print("k\tBA_train\tBA_test")
    for k, v in results.items():
        print(f"{k}\t{v['ba_train']:.3f}\t\t{v['ba_test']:.3f}")

if __name__ == "__main__":
    run_ex2()
