import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer



def average(scores):
    return np.mean(scores, axis=1, keepdims=True)


def maximization(scores):
    """Compute max score across models."""
    return np.max(scores, axis=1, keepdims=True)


def run_ex4(file_path='cardio.mat', model_type='knn', n_models=10,
            neighbors_range=(30, 120), contamination=None):
    # Load dataset
    mat = loadmat(file_path)
    print("Keys:", [k for k in mat.keys() if not k.startswith('__')])

    if 'X' in mat and 'y' in mat:
        X, y = mat['X'], mat['y'].ravel()
    else:
        keys = [k for k in mat.keys() if not k.startswith('__')]
        X, y = mat[keys[0]], mat[keys[1]].ravel()

    #0 (normal) and 1 (outlier)
    y = (y != 0).astype(int)

    #et contamination
    if contamination is None:
        contamination = np.mean(y)
    print(f"Contamination: {contamination:.4f}")

    #split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42, stratify=y
    )

    #normalize
    X_train, X_test = standardizer(X_train, X_test)

    #range
    neighbors = np.linspace(neighbors_range[0], neighbors_range[1], n_models).astype(int)
    print("Neighbors:", neighbors)

    train_scores, test_scores = [], []

    #train
    for k in neighbors:
        if model_type == 'knn':
            clf = KNN(n_neighbors=k, contamination=contamination)
        else:
            clf = LOF(n_neighbors=k, contamination=contamination)

        clf.fit(X_train)
        train_scores.append(clf.decision_scores_.reshape(-1, 1))
        test_scores.append(clf.decision_function(X_test).reshape(-1, 1))

        thr = np.quantile(test_scores[-1], 1 - contamination)
        y_pred = (test_scores[-1].ravel() >= thr).astype(int)
        ba = balanced_accuracy_score(y_test, y_pred)
        print(f"k={k} -> BA={ba:.3f}")

    #combine all model scores
    train_scores = np.hstack(train_scores)
    test_scores = np.hstack(test_scores)
    train_scores, test_scores = standardizer(train_scores, test_scores)
    avg_scores = average(test_scores)
    max_scores = maximization(test_scores)

    thr_avg = np.quantile(avg_scores, 1 - contamination)
    thr_max = np.quantile(max_scores, 1 - contamination)

    y_avg = (avg_scores.ravel() >= thr_avg).astype(int)
    y_max = (max_scores.ravel() >= thr_max).astype(int)

    ba_avg = balanced_accuracy_score(y_test, y_avg)
    ba_max = balanced_accuracy_score(y_test, y_max)

    print("\nFinal Ensemble Results:")
    print(f"Average combo -> BA = {ba_avg:.3f}")
    print(f"Max combo     -> BA = {ba_max:.3f}")


if __name__ == "__main__":
    print("KNN ensemble:")
    run_ex4('cardio.mat', model_type='knn')

    print("\nLOF ensemble:")
    run_ex4('cardio.mat', model_type='lof')
