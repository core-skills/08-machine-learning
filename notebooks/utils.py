
#
# Some functions to be used in the tutorial
#
# Developed by Debora Cristina Correa

import datetime
import pandas as pd
import matplotlib.pyplot as plt # for 2D plotting
import numpy as np
import seaborn as sns # plot nicely =)

from sklearn.base import clone

from sklearn.decomposition import PCA

def plot_decision_boundary(x_train, y_train, estimator):
    """Plot the decision boundary

    based on: http://scikit-learn.org/stable/auto_examples/semi_supervised/plot_label_propagation_versus_svm_iris.html

    Parameters
    ----------
    x_train: training set
    y_train: labels of the training set
    estimator: classifier, probability must be set as True
    """

    def make_meshgrid(x, y, h=.02):
        """Create a mesh of points to plot in

        Parameters
        ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

        Returns
        -------
        xx, yy : ndarray
        """
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        return xx, yy
    
    # cloning the estimator
    clf = clone(estimator)

    pca = PCA(n_components=2)
    x_train_pca = pca.fit_transform(x_train)
    clf.fit(x_train_pca, y_train)
    
    xx, yy = make_meshgrid(x_train_pca[:, 0], x_train_pca[:, 1])
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = clf.predict_proba(grid)[:, 1].reshape(xx.shape)
    
    # plotting the decision boundary
    f, ax = plt.subplots(figsize=(8, 6))

    sns.set(context="notebook", style="whitegrid",
            rc={"axes.axisbelow": False})

    contour = ax.contourf(xx, yy, probs, 25, cmap="RdBu",
                          vmin=0, vmax=1)

    ax_c = f.colorbar(contour)
    ax_c.set_label("$P(y = 1)$")
    ax_c.set_ticks([0, .25, .5, .75, 1])

    ax.scatter(x_train_pca[:,0], x_train_pca[:, 1], c=y_train, s=50,
               cmap="RdBu", vmin=-.2, vmax=1.2,
               edgecolor="white", linewidth=1)

    ax.set(aspect="equal",
           xlim=(-5, 5), ylim=(-5, 5),
           xlabel="$X_1$", ylabel="$X_2$")
    
    plt.show()
    