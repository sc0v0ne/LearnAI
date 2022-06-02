# Guide to Classification and Regression

<!-- MarkdownTOC -->

- Exploratory Data Analysis \(EDA\)
    - Why do we need univariate analysis?
    - Why do we need correlation analysis?
- Feature Engineering
- Model Implementation
- Model Evaluation
- Guide to Logistic Regression
- Terminology
- Classification Examples
- Regression Examples
- References

<!-- /MarkdownTOC -->

Linear regression is a good starting point when dealing with regression problems and can be considered the _baseline_ model. 

Linear regression finds the optimal linear relationship between independent variables and dependent variables in order to make predictions.

There are two main types:

- Simple regression (y = mx + b)
- Multivariable regression


## Exploratory Data Analysis (EDA)

EDA is essential to both investigate the data quality and reveal hidden correlations among variables.

1. Univariate Analysis

Visualize the data distribution using histogram for numeric variables and bar chart for categorical variables.

### Why do we need univariate analysis?

- Determine if dataset contains outliers

- Detrmine if we need data transformations or feature engineering

In this case, we found out that “expenses” follows a power law distribution, which means that log transformation is required as a step of feature engineering step, to convert it to normal distribution.

2. Multivariate Analysis

When thinking of linear regression, the first visualization technique that we can think of is scatterplot. 

By plotting the target variable against the independent variables using a single line of code `sns.pairplot(df)`, the underlying linear relationship becomes more evident.

3. Correlation Analysis

Correlation analysis examines the linear correlation between variable pairs which can be achieved by combining `corr()` function with `sns.heatmap()`. 

### Why do we need correlation analysis?

- To identify collinearity between independent variables — linear regression assumes no collinearity among independent features, so it is essential to drop some features if collinearity exists. 

- To identify independent variables that are strongly correlated with the target — strong predictors.


## Feature Engineering

EDA brought some insights of what types of feature engineering techniques are suitable for the dataset.

1. Log Transformation

We  found that the target variable  “expenses” is right skewed and follows a power law distribution. 

Since linear regression assumes linear relationship between input and output variable, it is necessary to use log transformation to “expenses” variable. 

As shown below, the data tends to be more normally distributed after applying `np.log2()`.

2. Encoding Categorical Variable

Another requirement of machine learning algorithms is to encode categorical variable into numbers.

Two common methods are one-hot encoding and label encoding. 


## Model Implementation

A simple linear regression y = b0 + b1x predicts relationship between one independent variable x and one dependent variable y. 

As more features/independent variables are introduced, it becomes multiple linear regression y = b0 + b1x1 + b2x2 + ... + bnxn, which cannot be easily plotted using a line in a two dimensional space.

Here we use `LinearRegression()` class from scikit-learn to implement the linear regression. 

We specify `normalize = True` so that independent variables will be normalized and transformed into same scale. 

Note that scikit-learn linear regression utilizes **Ordinary Least Squares** to find the optimal line to fit the data which means the line, defined by coefficients b0, b1, b2 … bn, minimizes the residual sum of squares between the observed targets and the predictions (the blue lines in chart). 


## Model Evaluation

Linear regression model can be qualitatively evaluated by visualizing error distribution. 

There are also quantitative measures such as MAE, MSE, RMSE and R squared.

1. Error Distribution

Here we use a histogram to visualize the distribution of error which should somewhat conform to a normal distribution. 

A non-normal error distribution may indicates that there is non-linear relationship that model failed to pick up or more data transformations are needed.

2. MAE, MSE, RMSE

All three methods measure the errors by calculating the difference between predicted values ŷ and actual value y, so the smaller the better. 

The main difference is that MSE/RMSE penalized large errors and are differentiable whereas MAE is not differentiable which makes it hard to apply in gradient descent. 

Compared to MSE, RMSE takes the square root which maintains the original data scale.

3. R Squared

R squared or _coefficient of determination_ is a value between 0 and 1 that indicates the amount of variance in actual target variables explained by the model. 

R squared is defined as 1 — RSS/TSS which is 1 minus the ratio between sum of squares of residuals (RSS) and total sum of squares (TSS). 

Higher R squared means better model performance.

In this case, a R squared value of 0.78 indicating that the model explains 78% of variation in target variable, which is generally considered as a good rate and not reaching the level of overfitting.


----------


## Guide to Logistic Regression

Logistic regression is a good starting point when dealing with classification problems and can be considered the _baseline_ model. 

Supervised learning refers to machine learning that is based on a training set of labeled examples. 

A supervised learning model trains on a dataset containing features that explain a target.

Here we review the following: 

- Logistic Regression (binary classification)

- Sigmoid Function

- Decision Boundaries for Multi-Class Problems

- Fitting Logistic Regression Models in Python

- Classification Error Metrics

- Error Metrics in Python


## Terminology

The parameters of the train function are called _hyperparameters_ such as iterations and learning rate which are set so that the train function can find parameters such as w and b.


----------



## Classification Examples

[End-to-End Machine Learning Workflow (Part 1)](https://medium.com/mlearning-ai/end-to-end-machine-learning-workflow-part-1-b5aa2e3d30e2)

[End-to-End Machine Learning Workflow (Part 2)](https://medium.com/mlearning-ai/end-to-end-machine-learning-workflow-part-2-e7b6d3fb1d53)

[Regression for Classification](https://towardsdatascience.com/regression-for-classification-hands-on-experience-8754a909a298)


[An End-to-End Machine Learning Project — Heart Failure Prediction](https://towardsdatascience.com/an-end-to-end-machine-learning-project-heart-failure-prediction-part-1-ccad0b3b468a?gi=498f31004bdf)

[One-vs-Rest and One-vs-One for Multi-Class Classification](https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/)


[Classification And Regression Trees for Machine Learning](https://machinelearningmastery.com/classification-and-regression-trees-for-machine-learning/)

[Building a Random Forest Classifier to Predict Neural Spikes](https://medium.com/@mzabolocki/building-a-random-forest-classifier-for-neural-spike-data-8e523f3639e1)

[KNN Algorithm for Classification and Regression: Hands-On With Scikit- Learn](https://cdanielaam.medium.com/knn-algorithm-for-classification-and-regression-hands-on-with-scikit-learn-4c5ec558cdba)


## Regression Examples

[A Beginner’s Guide to End to End Machine Learning](https://towardsdatascience.com/a-beginners-guide-to-end-to-end-machine-learning-a42949e15a47?gi=1736097101b9)

[A Practical Guide to Linear Regression](https://towardsdatascience.com/a-practical-guide-to-linear-regression-3b1cb9e501a6?gi=ba29357dcc8)

[A Practical Introduction to 9 Regression Algorithms](https://towardsdatascience.com/a-practical-introduction-to-9-regression-algorithms-389057f86eb9)



## References

[A Practical Guide to Linear Regression](https://towardsdatascience.com/a-practical-guide-to-linear-regression-3b1cb9e501a6?gi=ba29357dcc8)

[End-to-End Machine Learning Workflow](https://medium.com/mlearning-ai/end-to-end-machine-learning-workflow-part-1-b5aa2e3d30e2)

[Essential guide to Multi-Class and Multi-Output Algorithms in Python](https://satyam-kumar.medium.com/essential-guide-to-multi-class-and-multi-output-algorithms-in-python-3041fea55214)

[Supervised Machine Learning: Classification, Logistic Regression, and Classification Error Metrics](https://medium.com/the-quant-journey/supervised-machine-learning-classification-logistic-regression-and-classification-error-metrics-6c128263ac64?source=rss------artificial_intelligence-5)


[Five Regression Python Modules That Every Data Scientist Must Know](https://towardsdatascience.com/five-regression-python-modules-that-every-data-scientist-must-know-a4e03a886853)

[How to Transform Target Variables for Regression in Python](https://machinelearningmastery.com/how-to-transform-target-variables-for-regression-with-scikit-learn/)


[A Gentle Introduction to Multiple-Model Machine Learning](https://machinelearningmastery.com/multiple-model-machine-learning/)

