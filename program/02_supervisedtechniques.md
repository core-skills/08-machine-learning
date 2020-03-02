[Overview](./00_overview.md) | [ML Workflow](./01_mlworkflow.md) | [Supervised techniques](./02_supervisedtechniques.md) | [Model Evaluation I](./03_modelevaluationA.md)  | [Model Evaluation II](./04_modelevaluationB.md) | [Closeout](./05_closeout.md)

# Supervised techniques

| *90 min*  |
| --------- |

### Support Vector Machines (SVMs)

| *30 min*  |
| --------- |

SVMs are powerful discriminative classifiers formally defined by the concept of maximing a separating hyperplane. Given labeled training data, the algorithm outputs an optimal hyperplane which categorizes new examples.

SVMs use the kernel trick to project the data to another space by using the **kernel** in order to perform the (linear) classification. Indeed, kernels are the secret source that makes SVMs interesting ML techniques. We are going to explore three different types of kernels:

- Linear Kernel
- Radial Basis Function Kernel
- Polynomial Kernel

**Exercise**
Open [am1-iron-ore-dataset.ipynb](../notebooks/am1-iron-ore-dataset.ipynb) and go through the the exercises related to the SVM classification.

### Decision Trees and Random Forests

| *30 min*  |
| --------- |

Decition Trees and Random Forests are examples of information-based learning. It seeks for the most informative feature to split the dataset. The Random Forest classifier builds multiple decision trees and merges them together to get a more accurate and stable prediction (less proned to overfit).

**Exercise**
Open [am1-iron-ore-dataset.ipynb](../notebooks/am1-iron-ore-dataset.ipynb) and go through the the exercises related to the Random Forest classification.

### K-Nearest Neighbors (K-NN)

| *30 min*  |
| --------- |

In short, a new instance is classified in this algorithm by a majority vote according its neighbors. The instance is assigned to the most common class among its K nearest neighbors.

**Exercise**
Open [am1-iron-ore-dataset.ipynb](../notebooks/am1-iron-ore-dataset.ipynb) and go through the the exercises related to the KNN classification.

| *Lunch*  |
| --------- |