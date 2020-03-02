[Overview](./00_overview.md) | [ML Workflow](./01_mlworkflow.md) | [Supervised techniques](./02_supervisedtechniques.md) | [Model Evaluation I](./03_modelevaluationA.md)  | [Model Evaluation II](./04_modelevaluationB.md) | [Closeout](./05_closeout.md)

# The Machine Learning Workflow

| *80 min*  |
| --------- |

As always, whenever you have an issue related to data analysis and machine learning, go check scikit-learn's website [scikit-learn.org](https://scikit-learn.org/).

Documentation: 
[scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)

Examples:
[scikit-learn.org/stable/auto_examples/index.html](https://scikit-learn.org/stable/auto_examples/index.html)

Glossary:
[scikit-learn.org/stable/glossary.html](https://scikit-learn.org/stable/glossary.html)

Stack Overflow is a useful website to find solutions for when you get stuck with python and/or scikit-learn: [https://stackoverflow.com/questions/tagged/scikit-learn](https://stackoverflow.com/questions/tagged/scikit-learn).

## GitHub Preparations

| *5 min*  |
| --------- |

Please, install the environment included in the GitHub folder. There is a version for Mac (environment_mac.yml) and a version for Windows (environment_win.yml). 

Also please open the handouts for this morning session at [CORE_week6_early_morning.pptx]((../handouts/CORE_week6_early_morning.pptx))

## Review of the data science steps

| *20 min*  |
| --------- |

|Question: What are the main steps of the data analysis workflow?   |
| ------------------------------------------------------------------- |

Let's review the main steps in data analysis and understand how the ML techniques fit in this framework. 

1. Study the problem
2. Frame the questions
3. Collect/import data
4. Data exploration and preparation. This includes data cleaning, tyding, feature selection, feature extraction, dimensionality reduction.
5. Build model
6. Evaluate model
7. Report

Machine learning models are part of steps 5 and 6. We will discuss how they work today. 

|Question: What are the different types of data we can have?   |
| ------------------------------------------------------------------- |

This can include images, time series or structure data. There are ML techniques and data science methods designed to deal with each one of them. 

## Types of Machine Learning methods

| *20 min*  |
| --------- |

|Question: What is ML? How it differs from data mining and AI?    |
| ------------------------------------------------------------------- |

|Question: What are the different types of machine learning methods?   |
| ------------------------------------------------------------------- |

They include instance-based learning and model-based learning; online learning and batch learning; supervised, unsupervised and reinforcement learning. Let's discuss them. 

Don't forget to have a look at the Scikit-learn flowchart to help find the right method for the right problem and dataset: 
[scikit-learn.org/stable/tutorial/machine_learning_map/index.html](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)

## Exercise:

| *20 min*  |
| --------- |

| We will present several interesting (industry) scenarios. We would like you to discuss in group how they would formulate the ML problem in each scenario, and what are the inputs and outputs in each situation.|
| ------------------------------------------------------------------- |

## Supervised machine learning: general steps

| *5 min*  |
| --------- |

|Question: What are the steps of supervised ML methods?   |
| ------------------------------------------------------------------- |

1. Formulation of the problem
2. Data exploration and preparation
3. Evaluation of different models
4. Fine-tuning
5. Monitoring and maintenance

## Supervised machine learning: bias x variance

| *10 min*  |
| --------- |

|Question: What is underfitting and overfitting in ML?   |
| ------------------------------------------------------------------- |

Good machine learning models will be able to generalise well from the training examples and make good predictions in the future on new data (never seen) data. The bias x variance tradeoff is a fundamental concept in ML because it defines the situations known as underfitting and overfitting. These situations are responsible for the poor performance of ML models. In an underfitting situation, the model was not able to learn the training data, it is too simplistic (high bias). In the overfitting case, the model learned the training data too well, but it is too complex (high variance) to generalise to new examples. 

| *Coffee break*  |
| --------- |



