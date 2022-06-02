# AI Tips and Tricks

<!-- MarkdownTOC -->

- ML Tips
- ML Problem Solving
- Resources
- Robotics
- MLOps
    - Tracking Model Experiments
    - Managing Machine Learning Features
    - Observing and Monitoring Model in Production
    - What could go wrong after deployment?
- References

<!-- /MarkdownTOC -->


[Applied Machine Learning Process](../process/applied_ml.md)

[AI Checklist](../checklist/ai_checklist.md)

[Applied ML Checklist](../checklist/applied_ml_checklist.md)

[AutoML Tools](./automl_tools.md)


Asynchronous vs Event-Driven Architecture


## ML Tips

[Machine Learning Tips](./ml_tips.md)

[Computer Vision Tips](./cv_tips.md)

[How to Diagnose Overfitting and Underfitting?](../ml/diagnose_overfitting.md)

[How to Profile Memory Usage?](./memory_usage.md)


## ML Problem Solving

[How to solve machine learning problems in the real world?](https://www.kdnuggets.com/2021/09/solve-machine-learning-problems-real-world.html)

[Why you should care about debugging machine learning models?](https://www.oreilly.com/radar/why-you-should-care-about-debugging-machine-learning-models/)



## Resources

[Machine Learning Cheatsheet](https://ml-cheatsheet.readthedocs.io/en/latest/index.html)

[Machine Learning Guide](https://mclguide.readthedocs.io/en/latest/index.html)

[Getting Started with Machine Learning](https://machinelearningmastery.com/start-here/)



## Robotics

[Robotics Learning](https://www.rosroboticslearning.com/)

[Robotics Kinematics](https://robocademy.com/2020/04/21/robot-kinematics-in-a-nutshell/)

[Modern Robotics](https://modernrobotics.northwestern.edu/nu-gm-book-resource/foundations-of-robot-motion/#department)

[The Ultimate Guide to Jacobian Matrices for Robotics](https://automaticaddison.com/the-ultimate-guide-to-jacobian-matrices-for-robotics/)


## MLOps

Like DevOps, MLOps manages automated deployment, configuration, monitoring, resource management and testing and debugging.

Unlike DevOps, MLOps also might need to consider data verification, model analysis and re-verification, metadata management, feature engineering and the ML code itself.

But at the end of the day, the goal of MLOps is very similar. The goal of MLOps is to create a continuous development pipelines for machine learning models.

A pipeline that quickly allows data scientists and machine learning engineers to deploy, test and monitor their models to ensure that their models continue to act in the way they are expected to.

**Key ideas:** 

- Need to version code, data, and models
- Continuous delivery and continuous training

### Tracking Model Experiments

Unlike the traditional software development cycle, the model development cycle paradigm is different. 

A number of factors influence an ML model’s success in production. 

1. The outcome of a model is measured by its metrics such as an acceptable accuracy.

2. Many models and many ML libraries while tracking each experiment runs: metrics, parameters, artifacts, etc.

3. Data preparation (feature extractions, feature selection, standardized or normalized features, data imputations and encoding) are all important steps before the cleansed data lands into a feature store, accessible to your model training and testing phase or inference in deployment.

4. The choice of an ML framework for taming compute-intensive ML workloads: deep learning, distributed training, hyperparameter optimization (HPO), and inference.

5. The ability to easily deploy models in diverse environments at scale: part of web applications, inside mobile devices, as a web service in the cloud, etc.

### Managing Machine Learning Features

Feature stores address operational challenges. They provide a consistent set of data between training and inference. They avoid any data skew or inadvertent data leakage. They offer both customized capability of writing feature transformations, both on batch and streaming data, during the feature extraction process while training. And they allow request augmentation with historical data at inference, which is common in large fraud and anomaly detection deployed models or recommendation systems.

### Observing and Monitoring Model in Production

Data drift: As we mentioned above, our quality and accuracy of the model depends on the quality of the data. Data is complex and never static, meaning what the original model was trained with the extracted features may not be as important over time. Some new features may emerge that need to be taken into account. 

Model concept drift: Many practitioners refer to this as model decay or model staleness. When the patterns of trained models no longer hold with the drifting data, the model is no longer valid because the relationships of its input features may not necessarily produce the model’s expected prediction. Thus, its accuracy degrades.

Models fail over time: Models fail for inexplicable reasons: a system failure or bad network connection; an overloaded system; a bad input or corrupted request. Detecting these failures’ root causes early or its frequency mitigates user bad experience or deters mistrust in the service if the user receives wrong or bogus outcomes.

Systems degrade over load: Constantly being vigilant of the health of your dedicated model servers or services deployed is just as important as monitoring the health of your data pipelines that transform data or your entire data infrastructure’s key components: data stores, web servers, routers, cluster nodes’ system health, etc.

### What could go wrong after deployment?

What kind of problems Machine Learning applications might encounter over time.

- Changes in data distribution (data drifts): sudden changes in the features values.

- Model/concept drifts: how, why and when the performance of your model dropped.

- System performance: training pipelines failing, or taking long to run; very high latency...

- Outliers: the need to track the results and performances of a model in case of outliers or unplanned situations.

- Data quality: ensuring the data received in production is processed in the same way as the training data.


## References

[Considerations for Deploying Machine Learning Models in Production](https://towardsdatascience.com/considerations-for-deploying-machine-learning-models-in-production-89d38d96cc23)

[What Is MLOps And Why Your Team Should Implement It](https://medium.com/smb-lite/what-is-mlops-and-why-your-team-should-implement-it-b05b741cdf94)

[Design Patterns in Machine Learning for MLOps](https://towardsdatascience.com/design-patterns-in-machine-learning-for-mlops-a3f63f745ce4)

