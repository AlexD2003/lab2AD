import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from sklearn.linear_model import LinearRegression
np.random.seed(0)

def leverage_scores(X):
    #design matrix
    H = X @ inv(X.T @ X) @ X.T
    return np.clip(np.diag(H), 0, 1)

def make_points_1d(n_regular=50, n_high_x=10, n_high_y=10, n_high_both=10, a=2.0, b=1.0, mu=0.0, sigma=1.0):
    #small noise
    xr = np.random.normal(0, 1.0, n_regular)
    yr = a * xr + b + np.random.normal(mu, sigma * 0.2, n_regular)  # small noise

    #high-x variance
    xhx = np.random.normal(0, 5.0, n_high_x)
    yhx = a * xhx + b + np.random.normal(mu, sigma * 0.2, n_high_x)

    #high-y variance
    xhy = np.random.normal(0, 1.0, n_high_y)
    yhy = a * xhy + b + np.random.normal(mu, sigma * 5.0, n_high_y)

    #high both
    xhb = np.random.normal(0, 6.0, n_high_both)
    yhb = a * xhb + b + np.random.normal(mu, sigma * 6.0, n_high_both)

    xs = np.concatenate([xr, xhx, xhy, xhb])
    ys = np.concatenate([yr, yhx, yhy, yhb])
    labels = np.concatenate([
        np.zeros_like(xr),        #label 0
        np.ones_like(xhx) * 1,    #label 1
        np.ones_like(xhy) * 2,    #label 2
        np.ones_like(xhb) * 3     #label 3
    ])
    return xs.reshape(-1,1), ys, labels

def plot_leverage_1d(noise_sigma_values=[0.1, 1.0, 5.0]):
    fig, axs = plt.subplots(1, len(noise_sigma_values), figsize=(5 * len(noise_sigma_values), 4))
    if len(noise_sigma_values) == 1:
        axs = [axs]
    for ax, sigma in zip(axs, noise_sigma_values):
        Xx, y, groups = make_points_1d(sigma=sigma)
        X_design = np.hstack([np.ones((Xx.shape[0],1)), Xx])
        lev = leverage_scores(X_design)
        #plot
        scatter = ax.scatter(Xx.ravel(), y, c=groups, cmap='tab10', s=50, alpha=0.8, edgecolor='k')
        #highlight top leverages
        top_idx = np.argsort(-lev)[:6]
        ax.scatter(Xx.ravel()[top_idx], y[top_idx], facecolors='none', edgecolors='r', s=200, linewidths=2, label='Top leverage')
        ax.set_title(f'noise sigma = {sigma}')
        ax.set_xlabel('x'); ax.set_ylabel('y')
        ax.legend()
    plt.tight_layout()
    plt.show()

#y = a*x1 + b*x2 + c + eps
def make_points_2d(n_regular=150, n_outliers=30, a1=2.0, a2=-1.5, c=0.5, mu=0.0, sigma=1.0):
    #regular cluster
    Xr = np.random.normal(0, 1.0, (n_regular, 2))
    Yr = a1 * Xr[:,0] + a2 * Xr[:,1] + c + np.random.normal(mu, sigma * 0.2, n_regular)
    #outliers with varied x
    Xo1 = np.random.normal(0, 6.0, (int(n_outliers/2), 2))  # large x
    Yo1 = a1 * Xo1[:,0] + a2 * Xo1[:,1] + c + np.random.normal(mu, sigma * 6.0, int(n_outliers/2))
    Xo = np.vstack([Xo1])
    Yo = np.concatenate([Yo1])

    X = np.vstack([Xr, Xo])
    y = np.concatenate([Yr, Yo])
    labels = np.concatenate([np.zeros(n_regular), np.ones(len(Xo))])
    X_design = np.hstack([np.ones((X.shape[0],1)), X])
    lev = leverage_scores(X_design)
    return X, y, labels, lev

def plot_2d_leverage():
    X, y, labels, lev = make_points_2d()
    plt.figure(figsize=(6,5))
    plt.scatter(X[:,0], X[:,1], c=labels, cmap='tab10', s=50, edgecolor='k')
    top_idx = np.argsort(-lev)[:8]
    plt.scatter(X[top_idx,0], X[top_idx,1], facecolors='none', edgecolors='r', s=200, linewidths=2, label='Top leverage')
    plt.show()

if __name__ == "__main__":
    plot_leverage_1d([0.1, 1.0, 5.0])
    plot_2d_leverage()
