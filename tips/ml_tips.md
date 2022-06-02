# Machine Learning Tips

<!-- MarkdownTOC -->

- Avoid Using Different Library Versions
- What aspect ratio to use for line plots?
  - Calculating the aspect ratio
  - Best practices
- Run ML model training/evaluation with TMUX
- Watch your training and GPU resources
- Testing the online inference models
  - A/B test
- Monitoring the model
- References

<!-- /MarkdownTOC -->


## Avoid Using Different Library Versions

A mistake we might run into is to use different versions of the various exploited libraries during the train / test and deployment phase.

The risk of using different versions is to have unexpected behaviours which may lead to wrong predictions.

A possible solution to this problem could be to create a virtual environment and install all the necessary libraries, also specifying the versions to be used and then use this virtual environment both during the train/test phase and during the deployment phase.



## What aspect ratio to use for line plots?

One of the most overlooked aspects of creating charts is the use of correct aspect ratios. 

### Calculating the aspect ratio

The concept of banking to 45 degrees is used to have coherency between the information presented and information perceived. 

Thus, we need to make sure that the orientation of the line segments in the chart is as close as possible to a slope of 45 degrees.

Here, the median absolute slope banking method has been used to calculate the aspect ratio for the sunspots plot. ￼

The `ggthemes` package provides a function called bank_slopes() to calculate the aspect ratio of the plot which takes x and y values as the two arguments. The default method is the median absolute slope banking. 

### Best practices

- **Plotting multiple line graphs for comparison on a single chart:** The default aspect ratio works only if you do not plan to compare two different plots.

- **Comparing different line graphs from different charts:** Make sure the aspect ratio for each plot remains the same. Otherwise, the visual interpretation will be skewed. 

  1. Using incorrect or default aspect ratios: In this case, we choose the aspect ratios such that the plots end up being square-shaped.

  2. Calculating aspect ratios per plot: The best approach to compare the plots is to calculate the aspect ratios for each plot. 

**Time-series:** It is best to calculate the aspect ratio since some hidden information can be more pronounced when using the correct aspect ratio for the plot.


## Run ML model training/evaluation with TMUX

`tmux` can be used when you want to detach processes from their controlling terminals which allows remote sessions to remain active without beingvisible.


## Watch your training and GPU resources

```bash
  watch -n nvidia-smi
  nvtop
  gpustat
```


## Testing the online inference models

ML system testing is also more complex a challenge than testing manually coded systems, due to the fact that ML system behavior depends strongly on data and models that cannot be strongly specified a priori.

Figure: The Machine Learning Test Pyramid.

ML requires more testing than traditional software engineering.

### A/B test

To measure the impact of a new model, we need to augment the evaluation by running statistical A/B tests. 

In an A/B test, users are split into two distinct non-overlapping cohorts. To run an A/B test, the population of users must be split into statistically identical populations that each experience a different algorithm.


## Monitoring the model

Once a model has been deployed its behavior must be monitored. 

A model’s predictive performance is expected to degrade over time as the environment changes callef concept drift which occurs when the distributions of the input features or output target shift away from the distribution upon which the model was originally trained.

When concept drift has been detected, we need to retrain the ML model but detecting drift can difficult.

One strategy for monitoring is to use a metric from a deployed model that can be measured over time such as measuring the output distribution. The observed distribution can be compared to the training output distribution, and alerts can notify data scientists when the two quantities diverge.

Popular ML/AI deployment tools: TensorFlow Serving, MLflow, Kubeflow, Cortex, Seldon.io, BentoML, AWS SageMaker, Torchserve, Google AI.




## References

[Best practices in the deployment of AI models](https://nagahemachandchinta.medium.com/best-practices-in-the-deployment-of-ai-models-c929c3146416)

[Data Science Mistakes to Avoid: Data Leakage](https://towardsdatascience.com/data-science-mistakes-to-avoid-data-leakage-e447f88aae1c)

[10 Simple Things to Try Before Neural Networks](https://www.kdnuggets.com/2021/12/10-simple-things-try-neural-networks.html)

[What aspect ratio to use for line plots](https://towardsdatascience.com/should-you-care-about-the-aspect-ratio-when-creating-line-plots-ed423a5dceb3)

[Introduction to TensorFlow Probability (Bayesian Neural Network)](https://towardsdatascience.com/introduction-to-tensorflow-probability-6d5871586c0e)



