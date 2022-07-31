# Frequently Asked Questions

<!-- MarkdownTOC -->

- What is an AI Engineer
- What is the difference between AI and ML
- Do I need a Master’s Degree
- Problems with AI
- Recommended Tutorials and Books
- How to access Safari Online
- Can I learn AI from research papers
- How to ask an AI/ML question
- How to choose a performance metric
- How to Choose an ML Algorithm
- How to choose classification model
- Should I start learning ML by coding an algorithm from scratch
- Is image channels first or last
- How to share your work
- How to choose a Cloud Platform
- Common Questions on Encoding
  - What if I have hundreds of categories
  - What encoding technique is the best
  - What if I have a mixture of numeric and categorical data
- Common Questions on Normalization
  - Which Scaling Technique is Best
  - Should I Normalize or Standardize
  - Should I Standardize then Normalize
  - How Do I Handle Out-of-Bounds Values
- Using AI with Medical Images
- How to Develop a Chatbot
- How to Develop Recommender Systems
- Why are Robots not more common

<!-- /MarkdownTOC -->

## What is an AI Engineer

[Artificial Intelligence Engineering](https://www.sei.cmu.edu/our-work/artificial-intelligence-engineering/)

In simplest terms, an AI Engineer is a type of software engineer specializing in development of AI/ML applications. Thus, he/she needs to have a thorough understanding of the core software engineering concepts (SWEBOK) as well as the full software development life cycle for AI/ML applications which has some  differences.

> In a nutshell, when you create a program to solve an AI problem you are performing AI engineering. 


## What is the difference between AI and ML

In AI, we define things in terms of _agents_ (agent programs) and the _standard model_ (rational agents). Thus, we are performing _machine learning_ when the agent is a _computer_ versus Robot, AV, or UAV. 

Answer: When the agent is a computer. 

If you did not know the answer, you need to reevaluate your approach to learning AI. You need to concentrate on learning the key concepts, theory, and algorithms but avoid study of SOTA algorithms since the algorithms are constantly changing. 


## Do I need a Master’s Degree

If you are going to spend the time to study AI/ML then you might as well invest in an online degree which will greatly increase your career opportunities (and a requirement for most all AI/ML engineer positions).

The best approach would be to find several job postings that look interesting to you and see what skills and tools they require.

[How to Learn Machine Learning](https://aicoder.medium.com/how-to-learn-machine-learning-4ba736338a56)



## Problems with AI

Here are some facts to be aware of when learning AI/ML:

- AI is primarily a graduate level (MSCS) topic. 

- All students of AI need to know the history of AI. 

- The study of AI requires: 1) a healthy dose of skepticism and 2) the study of the history of AI which led to its downfall in the 1980s. 

There is currently a resurgence of misinformation and hype in AI research and software development which has led to the term "AI alchemy". 

- 80-85% (most likely higher) of AI projects fail (probably much higher for some areas such as AV, RL, and Robotics). Keep in mind these are for the most part experienced software engineers. 

- There is an ethical dilemma in AI research. 

  Almost 34% of AI researchers admit to manipulating results in a recent IEEE survey (most certainly higher since irreproduciblity is a known problem with AI research). 

- Engineering disciplines such as electrical engineering and civil engineering require a college degree, but software engineering (which includes AI/ML) does not require a degree (in the U.S.).  

  In fact, many commercial AI/ML applications have some of the same safety and other risk factors as other commercial engineering applications (perhaps more so with AV and Robotics). 

[IEEE Dataport](https://ieee-dataport.org)

[IEEE Share Your Data and Code](https://conferences.ieeeauthorcenter.ieee.org/get-published/share-your-data-and-code/)


## Recommended Tutorials and Books

[AI Learning Resources](https://aicoder.medium.com/ai-learning-resources-b49da21fd3b8)

Open source projects can be a good resume builder such as PyCaret, scikit-learn, etc. GitHub repos usually have a “good first issue” or "help wanted" tags on the Issues tab. It can be a bit of work but u should be able to find some open-source project that u can contribute to.

It is also good to do some small projects yourself to create a portfolio that you can share on your own GitHub repos, even if it is just to fork and add some features/enhancements yourself to some small ML repo that you find interesting.


## How to access Safari Online

If you have an .edu email account you can get free access to [Safari Online][^safari_online] which has some good books for beginners as well as advanced books on various AI/ML topics.

[Creating an Account](https://ecpi.libguides.com/SafariOReilly)

Some good books are “Artificial Intelligence with Python”, “Artificial Intelligence by Example”, and “Programming Machine Learning”.


## Can I learn AI from research papers

Research papers are not a good resource for learning a topic. The reader is assumed to already know the core theory and concepts covered in textbooks. Thus, the best approach to learning AI is a good textbook. 

AI is considered a graduate level topic in computer science, so there a lot of Math and CS prerequisites that are needed first to properly learn the core theory and concepts. Otherwise, it will be problematic at some point when you try to actually use AI to solve a real-world problem. 

In general, the results of AI research articles are irreproducible. In fact, there is a major problem with integrity in AI research right now. The further away you get from reputable publications such as IEEE, the worse it gets.  I have seen numerous articles that I have no idea how they were ever published (major mistakes and errors). When learning AI, you need a healthy dose of skepticism in what you read, especially research articles. This is all discussed in the Russell and Norivg and other graduate textbooks. 

[Is My Model Really Better?](https://towardsdatascience.com/is-my-model-really-better-560e729f81d2)


## How to ask an AI/ML question

Briefly describe the following (1-2 sentences per item):

1. Give some description of your background and experience.

2. Describe the problem.

3. Describe the dataset in detail and be willing to share your dataset.

4. Describe any data preparation and feature engineering steps that you have done.

5. Describe the models that you have tried. 

6. Favor text and tables over plots and graphs.

7. Avoid asking users to help debug your code. 

[How to ask an AI/ML question](https://aicoder.medium.com/how-to-ask-an-ai-ml-question-6cfddaa75bc9)

In AI and CS, you should always be able to describe in a few sentences what you are doing and why you are doing it. It is the first step in defining an AI problem. Also, there is a category and/or terminology for everything we do in CS. It always applies whether you are doing research or working on a simple problem. If you have not taken the time to think about the problem and put it into writing then u really don’t know what you are doing, do you?



## How to choose a performance metric

[Machine Learning Performance Metrics](./ml/performance_metrics.md) 



## How to Choose an ML Algorithm

First, remember to take a data-centric approach, so avoid asking “what models should I use”. Thus, the first step in ML process would be to perform EDA to understand the properties of your model such as balanced (classification) or Gaussian (regression).

Concentrate on learning the key concepts such as data preparation, feature engineering, model selection, sklearn and tflow and pipelines, tflow Dataset class, etc. It also would be good to work through a few end-to-end classification/regression examples to get an idea for some of the steps involved.


There are some applied AI/ML processes and techniques given in Chapter 19 of the following textbooks that include a procedure for model selection:

S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 4th ed. Upper Saddle River, NJ: Prentice Hall, ISBN: 978-0-13-604259-4, 2021.

E. Alpaydin, Introduction to Machine Learning, 3rd ed., MIT Press, ISBN: 978-0262028189, 2014.

J Brownlee also describes an “Applied Machine Learning Process” that I have found most helpful in practice. 

I have some notes and checklists that I have created concerning the applied AI process in general. The process is  similar to the approach of SWE best practices given in SWEBOK. 


[Getting Started with AI](https://medium.com/codex/getting-started-with-ai-13eafc77ac8e)

[LearnML](https://github.com/codecypher/LearnML)


[An End-to-End Machine Learning Project — Heart Failure Prediction](https://towardsdatascience.com/an-end-to-end-machine-learning-project-heart-failure-prediction-part-1-ccad0b3b468a?gi=498f31004bdf)

[A Beginner’s Guide to End to End Machine Learning](https://towardsdatascience.com/a-beginners-guide-to-end-to-end-machine-learning-a42949e15a47?gi=1736097101b9)

[End-to-End Machine Learning Workflow](https://medium.com/mlearning-ai/end-to-end-machine-learning-workflow-part-1-b5aa2e3d30e2)



## How to choose classification model

What do you do when you run a neural network that's barely optimizing (10-20%) over the baseline?

For more context, I am training a four layer neural network whose inputs are 2 x size 100 embeddings and the output is a boolean. The classes in each of my dataset are equally distributed (50% true, 50% false). As things stand, my model can detect falses with 75% accuracy and trues with just over 50%.

-----

Obviously, the model is not working since 50% accuracy is the lowest value which is the same as random guessing (coin toss). 

First, plot the train/val loss per epoch of any models being trained to see what is happening (over/under-fitting) but I always dismiss any models with that low of a baseline accuracy (too much work). 

Next, try to find a few well-performing models with good baseline results for further study. Pretty sure NN will not be the best model (usually XGBoost and Random Forest).

For classification, I always evaluate the following models:

- Logistic Regression
- Naive Bayes
- AdaBoost
- kNN
- Random Forest
- Gradient Boosting
- SGD
- SVM
- Tree

I would start by using AutoML tools (Orange, AutoGluon, etc.) or write a test harness to get a baseline on many simpler models.  AutoML tools can perform much of the feature engineering that is needed for the different models (some models require different data prep than others) which is helpful. The tools can also quickly perform PCA and other techniques to help with feature engineering and dimensionality reduction. 

I would first get a baseline on 10-20 simpler models first before trying NN. Even then, I would use tools such as SpaCy to evaluate pretrained NN models. 

Only if all those models failed miserably would I then try to roll my own NN model. In general, you want to find a well-performing model that requires the least effort (hypertuning) - Occam's Razor. Most any model can be forced to fit a dataset using brute-force which is not the correct approach for AI.  

I have lots of notes on text classification/sentiment analysis and NLP data preparation in the "nlp” folder of my repo in nlp.md and nlp_dataprep.md. 

There are a lot of steps to text preprocessing in which mistakes can be made and it is often trial-and-error, so I prefer using AutoML to obtain baselines first before spending a lot of effort in coding. 

https://github.com/codecypher/LearnAI



## Should I start learning ML by coding an algorithm from scratch

[How to Learn Machine Learning](https://aicoder.medium.com/how-to-learn-machine-learning-4ba736338a56)

I would recommend using Orange, AutoGluon, PyCaret and similar tools to evaluate many models (10-20) on your dataset. The tools will also help detect any issues with the data as well as perform the most common transforms needed for the various algorithms. Then, select the top 3 for further study. 

AutoML tools are the future of AI, so now would be a good time to see how they work rather than spend a lot of time coding the wrong algorithm from scratch which is a common beginner mistake. In short, you need to learn a data-centric approach.

The rule of thumb is that a deep learning model should be your last choice (Occam's Razor). 


## Is image channels first or last

A huge gotcha with both PyTorch and Keras. Actually, you need to sometimes need to to watch out when running code between OS (NHWC vs NCHW). I spent a long time tracking down an obscure error message between Linux and macOS that turned out to be the memory format.

[PyTorch Channels Last Memory Format Performance Optimization on CPU Path](https://gist.github.com/mingfeima/595f63e5dd2ac6f87fdb47df4ffe4772)

[Change Image format from NHWC to NCHW for Pytorch](https://stackoverflow.com/questions/51881481/change-image-format-from-nhwc-to-nchw-for-pytorch)

[A Gentle Introduction to Channels-First and Channels-Last Image Formats](https://machinelearningmastery.com/a-gentle-introduction-to-channels-first-and-channels-last-image-formats-for-deep-learning/)


[Learning to Resize in Computer Vision](https://keras.io/examples/vision/learnable_resizer/)


## How to share your work

Your are always welcome to share your work in the following Discord AI forum channels of the SHARE KNOWLEDGE Section: #share-your-projects, #share-datasets, and #our-accomplishments. 

A key ML skill needed is deployment. There are several options available but the simplest method to try first would be streamlit or deta cloud services.



## How to choose a Cloud Platform

[Comparison of Basic Deep Learning Cloud Platforms](https://aicoder.medium.com/comparison-of-basic-deep-learning-cloud-platforms-9a4b69f44a46)

[Colaboratory FAQ](https://research.google.com/colaboratory/faq.html#resource-limits)



## Common Questions on Encoding

This section lists some common questions and answers when encoding categorical data.

### What if I have hundreds of categories

What if I concatenate many one-hot encoded vectors to create a many-thousand-element input vector?

You can use a one-hot encoding up to thousands and tens of thousands of categories. Having large vectors as input sounds intimidating but the models can usually handle it.

### What encoding technique is the best

This is impossible to answer. The best approach would be to test each technique on your dataset with your chosen model and discover what works best.

### What if I have a mixture of numeric and categorical data

What if I have a mixture of categorical and ordinal data?

You will need to prepare or encode each variable (column) in your dataset separately then concatenate all of the prepared variables back together into a single array for fitting or evaluating the model.

Alternately, you can use the `ColumnTransformer` to conditionally apply different data transforms to different input variables.


## Common Questions on Normalization

This section lists some common questions and answers when scaling numerical data.

### Which Scaling Technique is Best

This is impossible to answer. The best approach would be to evaluate models on data prepared with each transform and use the transform or combination of transforms that result in the best performance for your data set and model.

### Should I Normalize or Standardize

Whether input variables require scaling depends on the specifics of your problem and of each variable.

If the distribution of the values is normal, it should be standardized. Otherwise, the data should be normalized.

The data should be normalized whether the range of quantity values is large (10s, 100s, ...) or small (0.01, 0.0001, ...).

If the values are small (near 0-1) and the distribution is limited (standard deviation near 1) you might be able to get away with no scaling of the data.

Predictive modeling problems can be complex and it may not be clear how to best scale input data.

If in doubt, normalize the input sequence. If you have the resources, explore modeling with the raw data, standardized data, and normalized data and see if there is a difference in the performance of the resulting model.

### Should I Standardize then Normalize

Standardization can give values that are both positive and negative centered around zero.

It may be desirable to normalize data after it has been standardized.

Standardize then Normalize may be a good approach if you have a mixture of standardized and normalized variables and would like all input variables to have the same minimum and maximum values as input for a given algorithm such as an algorithm that calculates distance measures.

### How Do I Handle Out-of-Bounds Values

You may normalize your data by calculating the minimum and maximum on the training data.

Later, you may have new data with values smaller or larger than the minimum or maximum respectively.

One simple approach to handling this may be to check for out-of-bound values and change their values to the known minimum or maximum prior to scaling. Alternately, you can estimate the minimum and maximum values used in the normalization manually based on domain knowledge.



## Using AI with Medical Images

Small and imbalanced datasets are common in medical applications. However, it is still considered an open research problem in CS. Thus, there is not standard “recipe” for data prep. Just some heuristics that people have come up with. So u will need to do some research to justify your final choice of data prep techniques, especially for medical datasets. 

At least one of the articles discusses X-ray images which may have some references that are helpful (not sure). I would try searching on arxiv.org for “Survey” articles that would list some peer-reviewed journal articles on the type of images that u are working with.

Resampling is just one approach to balance a dataset but it is an advanced concept, so you need to have a thorough understanding of the core ML concepts. Otherwise, your results will be suspect. 

I have some notes on “Dataset Issues” that may help get you started for structured datasets. However, the approach is different for image datasets.


## How to Develop a Chatbot

Chatbots are better to use pretrained model and software. You can take a look at Moodle and Rasa which are popular. There is also an example using NLTK that claims to be somewhat accurate. 

[How to Create a Moodle Chatbot Without Any Coding?](https://chatbotsjournal.com/how-to-create-a-moodle-chatbot-without-any-coding-3d08f95d94df)

[Building a Chatbot with Rasa](https://towardsdatascience.com/building-a-chatbot-with-rasa-3f03ecc5b324)

[Python Chatbot Project – Learn to build your first chatbot using NLTK and Keras](https://data-flair.training/blogs/python-chatbot-project/)



## How to Develop Recommender Systems

This article give a high level overview of recommender systems which may give some ideas for approaches. You can also track web browser activity in various ways to build a custom dataset but you should be able to find a toy dataset to work with initially once you decide on an approach.  

Once you decide on an approach (there are many such collaborative filtering for example), you should be able to determine the type of dataset that you need which will then allow to find a toy dataset to use for experimentation and prototyping since it will most likely take a lot of time and effort to build a custom dataset for many of the approaches such as web browser tracking. 

Then, you should have a toy dataset on which you can use some AutoML tools to evaluate many different models and be able to narrow the choices to just a few models.


[Inside recommendations: how a recommender system recommends](https://www.kdnuggets.com/inside-recommendations-how-a-recommender-system-recommends.html/)

[Recommender System using Collaborative Filtering in Pyspark](https://angeleastbengal.medium.com/recommender-system-using-collaborative-filtering-in-pyspark-b98eab2aea75?source=post_page-----b98eab2aea75-----------------------------------)



## Why are Robots not more common

Robot soccer is one type of classic robotics toy problem, often involving multiagent reinforcement learning (MARL). 

In robotics, there are a lot of technical issues (mainly safety related) involved besides the biomechanics of walking. 

There has been limited success for certain application areas such as the robot dog and of course robotic kits for arduino and raspberry pi (for hobbyists) but more practical applications still seem to be more elusive. 

In general, it costs a lot of money for R&D in robotics and it takes a long time for ROI. For example, openai recently disbanded its robotics division dues to lack of data and capital. Perhaps more interesting is the lack of a proper dataset which is needed for real-world robotics applications in the wild.


[OpenAI disbands its robotics research team](https://venturebeat.com/2021/07/16/openai-disbands-its-robotics-research-team/)

[Boston Dynamics now sells a robot dog to the public, starting at $74,500](https://arstechnica.com/gadgets/2020/06/boston-dynamics-robot-dog-can-be-yours-for-the-low-low-price-of-74500/)



[^safari_online]: https://www.oreilly.com/member/login/?next=%2Fapi%2Fv1%2Fauth%2Fopenid%2Fauthorize%2F%3Fclient_id%3D235442%26redirect_uri%3Dhttps%3A%2F%2Flearning.oreilly.com%2Fcomplete%2Funified%2F%26state%3DC3l2tLIMbQr0lpQKLDHucVJomOkg52rX%26response_type%3Dcode%26scope%3Dopenid%2Bprofile%2Bemail&locale=en "Safari Online"
