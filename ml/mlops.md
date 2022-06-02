# MLOps

<!-- MarkdownTOC -->

- AI SDLC
- What is a Modeling Pipeline
- References

<!-- /MarkdownTOC -->

## AI SDLC

In general, the ML model process involves eight stages which may also include data collection and/or data labeling [1]:

1. Data preparation
2. Feature engineering
3. Model design
4. Model training and optimization
5. Model evaluation
6. Model deployment
7. Model serving
8. Model monitoring


The software development lifecycle (SDLC) of an AI project can be divided into six stages [2]:

1. **Problem definition:** The formative stage of defining the scope, value definition, timelines, governance, resources associated with the deliverable.

2. **Dataset Selection:** This stage can take a few hours or a few months depending on the overall data platform maturity and hygiene. Data is the lifeblood of ML, so getting the right and reliable datasets is crucial.

3. **Data Preparation:** Real-world data is messy. Understanding data properties and preparing properly can save endless hours down the line in debugging.

4. **Design:** This phase involves feature selection, reasoning algorithms, decomposing the problem, and formulating the right model algorithms.

5. **Training:** Building the model, evaluating with the hold-out examples, and online experimentation. 

6. **Deployment:** Once the model is trained and tested to verify that it met the business requirements for model accuracy and other performance metrics, the model is ready for deployment. There are two common approaches to deployment of ML models to production: embed models into a web server or offload the model to an external service. Both ML model serving approaches have pros and cons.

7. **Monitoring:** This is the post-deployment phase involving observability of the model and ML pipelines, refresh of the model with new data, and tracking success metrics in the context of the original problem.


Serving: Model serving refers to the use of a platform to deploy ML models at massive scale. Examples: Seldon, KFServing, and Ray Serve.

Monitoring: This is the post-deployment phase involving observability of the model and ML pipelines, refresh of the model with new data, and tracking success metrics in the context of the original problem. Key items to monitor are:  model drift, data drift, model failure, and system performance. Examples: Evidently.ai, Arize.ai, Arthur.ai, Fiddler.ai, Valohai.com, or whylabs.ai.


The two most common architectures for ML model serving are:

1. **Precomputed Model Prediction:** This is one of the earliest used and simplest architecture for serving machine learning models. It is an indirect method for serving the model, where we precompute predictions for all possible combinations of input variables and store them in a database. This architecture is generally used in recommendation systems — recommendations are precomputed and stored and shown to the user at login.

2. **Microservice Based Model Serving:** The model is served independently of the application, and predictions are provided in real-time as per request. This type of architecture provides flexibility in terms of model training and deployment.



## What is a Modeling Pipeline

A _pipeline_ is a linear sequence of data preparation options, modeling operations, and prediction transform operations [3].

A pipeline allows the sequence of steps to be specified, evaluated, and used as an atomic unit.

**Pipeline:** A linear sequence of data preparation and modeling steps that can be treated as an atomic unit.


The first example uses data normalization for the input variables and fits a logistic regression model:

[Input], [Normalization], [Logistic Regression], [Predictions]


The second example standardizes the input variables, applies RFE feature selection, and fits a support vector machine.

[Input], [Standardization], [RFE], [SVM], [Predictions]


A pipeline may use a data transform that configures itself automatically, such as the RFECV technique for feature selection.

- When evaluating a pipeline that uses an automatically-configured data transform, what configuration does it choose? 

- When fitting this pipeline as a final model for making predictions, what configuration did it choose?

**The answer is, it does not matter.**

We are not concerned about the specific internal structure or coefficients of the chosen model.

We can inspect and discover the coefficients used by the model as an exercise in analysis, but it does not impact the selection and use of the model.

This same answer generalizes when considering a modeling pipeline.

We are not concerned about which features may have been automatically selected by a data transform in the pipeline. 

We are also not concerned about which hyperparameters were chosen for the model when using a grid search as the final step in the modeling pipeline.

The pipeline allows us as machine learning practitioners to move up one level of abstraction and be less concerned with the specific outcomes of the algorithms and more concerned with the capability of a sequence of procedures.

It is a shift in thinking that may take some time to get used to.



## References

[1] [What Is MLOps And Why Your Team Should Implement It](https://medium.com/smb-lite/what-is-mlops-and-why-your-team-should-implement-it-b05b741cdf94)

[2] [AI Checklist](https://towardsdatascience.com/the-ai-checklist-fe2d76907673)

[3] [A Gentle Introduction to Machine Learning Modeling Pipelines](https://machinelearningmastery.com/machine-learning-modeling-pipelines/)


[4] [Automate Machine Learning Workflows with Pipelines in Python and scikit-learn](https://machinelearningmastery.com/automate-machine-learning-workflows-pipelines-python-scikit-learn/)

[5] [Recursive Feature Elimination (RFE) for Feature Selection in Python](https://machinelearningmastery.com/rfe-feature-selection-in-python/)


[Managing Machine Learning Lifecycles with MLflow](https://kedion.medium.com/managing-machine-learning-lifecycles-with-mlflow-f230a03c4803)

[MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)

[Build an Anomaly Detection Pipeline with Isolation Forest and Kedro](https://towardsdatascience.com/build-an-anomaly-detection-pipeline-with-isolation-forest-and-kedro-db5f4437bfab)



