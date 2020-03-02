[Overview](./00_overview.md) | [ML Workflow](./01_mlworkflow.md) | [Supervised techniques](./02_supervisedtechniques.md) | [Model Evaluation I](./03_modelevaluationA.md)  | [Model Evaluation II](./04_modelevaluationB.md) | [Closeout](./05_closeout.md)

# Evaluating the models

| *60 min*  |
| --------- |

The performance assessment of the classifiers is extremely important in practice, as this provide insights of how the classifier performs with new data, in which me measure the generalisation error.

In this session, we will Recap/introduce fundamental concepts when evaluating the performance of a supervised ML model. 

- Training/val/test datasets
- Cross-validation 
- Variance x bias tradeoff 
- The learning curve
- The ROC curve
- What does it mean a ROC curve    
- Recall/Precision
- Confusion matrix
- Things to do in an overfitting or underfitting scenario
- Parameters vs hyper-parameters

**Exercise 1**
Open [pm1-performance-evaluation.ipynb](../notebooks/pm1-performance-evaluation.ipynb). It contains nice information about the evaluation metrics along with their implementation in Python. Go through them and do the proposed exercises related.

**Exercise 2**
Compare the evaluation results for the different classifiers with the previous K-NN classifier. Could you find some insights about the performance of both classfiers? 


## Imbalanced datasets
| *30 min*  |
| --------- |

|Question: Is accuracy a good performance indicator? Why?  |
| ------------------------------------------------------------------- |

**Exercise**
Open and explore [pm2-imbalanced-datasets.ipynb](../notebooks/pm2-imbalanced-datasets.ipynb). There are notes about imbalanced datasets and ideas about how to proper evaluate them. Also, there are questions and exercises for you to try. 

**Discussion**
Now let's discuss:

- What is the problem of confusion matrices and accuracy measures for imbalanced datasets?
- What does the evaluation indicate when precision and recall scores are so different, and also different from the accuracy score? Why recall score is so small in the example of the notebook?
- What does it mean an AUC equal 0.5?
- What are some strategies to deal with imbalanced datasets?
- 
| *Coffee break*  |
| --------- |
