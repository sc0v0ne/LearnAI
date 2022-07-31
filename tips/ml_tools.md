# Machine Learning Tools

<!-- MarkdownTOC levels=1,2 -->

- How to choose an ML framework
- Data Exploration Tools
- Feature Engineering Tools
- Computer Vision
- Deep Learning Tools
- ML Libraries
- NLP Pretrained Models
- NLP Libraries
- Time Series
- Pretrained Model Repos
- Python Libraries
- References

<!-- /MarkdownTOC -->

Here is a list of ML tools that I have found to be helpful for AI engineering.

For items without links, see **Github Lists**. 

[GitHub Lists](https://github.com/codecypher?tab=stars)

[Github Student Developer Pack](https://education.github.com/pack)


[HuggingFace Spaces](https://huggingface.co/spaces/launch)


## How to choose an ML framework

[Keras vs PyTorch for Deep Learning](https://towardsdatascience.com/keras-vs-pytorch-for-deep-learning-a013cb63870d)


## Data Exploration Tools

- Orange
- DataPrep
- Bamboolib
- TensorFlow Data Validation
- Great Expectations

NOTE: It is best to install the Orange native executable on your local machine rather than install using anaconda and/or pip.

### Tutorials

[Orange Docs](https://orangedatamining.com/docs/)

[A Great Python Library: Great Expectations](https://towardsdatascience.com/a-great-python-library-great-expectations-6ac6d6fe822e)


## Feature Engineering Tools

There are many tools that will help you in automating the entire feature engineering process and producing a large pool of features in a short period of time for both classification and regression tasks.

- Feature-engine
- Featuretools
- AutoFeat

### Tutorials

[The Only Web Scraping Tool you need for Data Science](https://medium.com/nerd-for-tech/the-only-web-scraping-tool-you-need-for-data-science-f388e2afa187)



## Computer Vision

- ageitgey/face_recognition

### OpenCV

[OpenCV](https://docs.opencv.org/master/d9/df8/tutorial_root.html)

OpenCV is a huge open-source library for computer vision, machine learning, and image processing. 

OpenCV supports a wide variety of programming languages like Python, C++, Java, etc. 

OpenCV can process images and videos to identify objects, faces, or eve handwriting.

### openpilot

The openpilot repo is an open-source driver assistance system that performs the functions of Adaptive Cruise Control (ACC), Automated Lane Centering (ALC), Forward Collision Warning (FCW) and Lane Departure Warning (LDW) for a growing variety of supported car makes, models, and model years. 

However, you will need to buy their product and install it on your car and it is not completely DIY but reduces the effort.


## Deep Learning Tools

- MXNet

### Hydra

[Hydra](https://hydra.cc/docs/intro/) is an open-source Python framework that simplifies the development of research and other complex applications. 

The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. 

The name Hydra comes from its ability to run multiple similar jobs - similar to a Hydra with multiple heads.

Hydra provides a configuration file for the entire experiment. We can have different parameters to be set. It can be very helpful when we want to share our code with someone else or run the experiments on a different machine. 

Hydra provides the flexibility to set the desired configurations such as learning rate, model hidden layer sizes, epochs, data set name, etc. without exposing someone to make changes to the actual code.

### H5py

[H5py](https://docs.h5py.org/en/stable/quick.html) can be used to store all the intermediate loss values in a dictionary mapped to appropriate key which can be loaded to be reused as a python code.

### Loguru

[Loguru](https://loguru.readthedocs.io/en/stable/api/logger.html) provides the functionality of a logger to log configuration, experiment name, and other training-related data which is helpful when we do multiple experiments and want to distinguish the results of different experiments. Thus, if we log the actual configuration as well as the results, then it is easier to map the appropriate setting to the outputs.

### Pickle

Pickle can be used to save and load the python classes or PyTorch models for reuse. We can pickle the objects and load it in future to save the time for preprocessing.

### Pipreqs

[Pipreqs](https://pypi.org/project/pipreqs/) is useful when we want to port our code to a different machine and install all the dependencies. It helps us in creating a list of python dependencies along with the versions that our current working code is using and save them in a file that can be easily installed anywhere else.

### Tqdm

When used with a loop (here we use with a loop over a torch.utils.data.DataLoader object), [Tqdm](https://tqdm.github.io/) provides a viewe of time per gradient step or epoch which can help us to set our logging frequency of different results or saving the model or get an idea to set the validation intervals.

### Tutorials

[Are You Still Using Virtualenv for Managing Dependencies in Python Projects?](https://towardsdatascience.com/poetry-to-complement-virtualenv-44088cc78fd1)

[3 Tools to Track and Visualize the Execution of Your Python Code](https://www.kdnuggets.com/2021/12/3-tools-track-visualize-execution-python-code.html)



## ML Libraries

- Miniforge with mamba (mambaforge)
- Kedro
- ONNX
- openai/gym
- poetry
- pyenv
- PyMC
- Snap ML
- Streamlit

### Snap ML

[Snap ML](https://www.zurich.ibm.com/snapml/)

Snap ML is a library that provides high-speed training of popular machine learning models on modern CPU/GPU computing systems

[This Library is 30 Times Faster Than Scikit-Learn](https://medium.com/@irfanalghani11/this-library-is-30-times-faster-than-scikit-learn-206d1818d76f)

[IBM Snap ML Examples](https://github.com/IBM/snapml-examples)


### Tutorials

[Introduction to OpenCV](https://www.geeksforgeeks.org/introduction-to-opencv/)

[OpenCV Python Tutorial](https://www.geeksforgeeks.org/opencv-python-tutorial/)

[How to start contributing to open-source projects](https://towardsdatascience.com/how-to-start-contributing-to-open-source-projects-41fcfb654b2e)

[A Gentle Introduction to Bayesian Belief Networks](https://machinelearningmastery.com/introduction-to-bayesian-belief-networks/)

[Building DAGs with Python](https://mungingdata.com/python/dag-directed-acyclic-graph-networkx/)

[bnlearn](https://github.com/erdogant/bnlearn)



## NLP Pretrained Models

- Polyglot
- SpaCy
- GenSim
- Pattern
- clean-text

[spaCy](https://github.com/explosion/spaCy) is a library for advanced Natural Language Processing in Python and Cython.

spaCy is built on the very latest research and was designed from day one to be used in real products.


## NLP Libraries

- Flair
- TextBlob
- VADER

The article [7] covers five useful Python recipes for your next NLP projects:

- Check metrics of text data with textstat
- Misspelling corrector with pyspellchecker
- Next word prediction with next_word_prediction
- Create an effective Word Cloud
- Semantic similarity analysis

Semantic similarity analysis

As opposed to _lexicographical similarity_, _semantic similarity_ measures the likeness of documents/sentences/phrases based on their meaning. 

The most effective methodology is to use a powerful transformer to encode sentences, get their embeddings and then use cosine similarity to calculate their distance/similarity score.

Calculating the cosine distance between two embeddings gives us the similarity score which is widely used in information retrieval and text summarization such as extract top N most similar sentences from multiple documents. 

The similarity scores can also be used to reduce the dimensionality and to find similar resources.



## Time Series

- statsmodels
- stumpy
- AutoTS
- Darts
- TsFresh



## Pretrained Model Repos

[Model Zoo](https://modelzoo.co/)

[TensorFlow Hub](https://tfhub.dev/)

[TensorFlow Model Garden](https://github.com/tensorflow/models/tree/master/official)

[Hugging Face](https://github.com/huggingface)

[PyTorch Hub](https://pytorch.org/hub/)

[Papers with Code](https://paperswithcode.com/)



## Python Libraries

- dateutil
- Pipreqs
- Poetry
- tqdm
- urllib3

- The Algorithms - Python
- vinta/awesome-python
- josephmisiti/awesome-machine-learning


### Jupyterlab

[JupyterLab](https://github.com/jupyterlab/jupyterlab) is the next-generation user interface for Project Jupyter offering all the familiar building blocks of the classic Jupyter Notebook (notebook, terminal, text editor, file browser, rich outputs, etc.) in a flexible and powerful user interface. JupyterLab will eventually replace the classic Jupyter Notebook.

Jupyterlab has an updated UI/UX with a tab interface for working with multiple files and notebooks.

Since Jupyter is really a web server application, it runs much better on a remote server. 

I currently have Jupyterlab installed and running as a Docker container on a VM droplet which runs much better than on my local machine. The only issue is that my VM only has 4GB memory. However, I have had great success so far using Jupyterlab and Modin with notebooks that I am unable to run on my local machine with 32GB memory (out of memory issues) without any performance issues.

If you do not have cloud server of your own, a nice alternative is [Deepnote](https://deepnote.com). The free tier does not offer GPU access but it does offer a shared VM with 24GB of memory running a custom version of Jupyterlab which I have found more useful than Google Colab Pro. It is definitely worth a try. 

### Modin

[Modin](https://github.com/modin-project/modin) is a drop-in replacement for pandas. 

While pandas is single-threaded, Modin lets you speed up your workflows by scaling pandas so it uses all of your cores. 

Modin works especially well on larger datasets where pandas becomes painfully slow or runs out of memory.

Using modin is as simple as replacing the pandas import:

```py
  # import pandas as pd
  import modin.pandas as pd
```

I have a sample [notebook](../python/book_recommender_knn.ipynb) that demonstrates using modin. 

Since Modin is still under development, I do experience occasional warning/error messages but everything seems to be working. However, the developers seem to be quick to answer questions and provide assistance in troubleshooting issues. Highly recommend trying it out. 


### Pickle

Pickle can be used to save and load the python classes or PyTorch models for reuse.

### PySpark

[Getting Started](https://spark.apache.org/docs/latest/api/python/getting_started/index.html)

PySpark is an interface for Apache Spark in Python. It not only allows you to write Spark applications using Python APIs, but also provides the PySpark shell for interactively analyzing your data in a distributed environment. PySpark supports most of Spark’s features such as Spark SQL, DataFrame, Streaming, MLlib (Machine Learning) and Spark Core.

### Debugging Tools

- heartrate
- Loguru
- snoop




## References

[All Top Python Libraries for Data Science Explained](https://towardsdatascience.com/all-top-python-libraries-for-data-science-explained-with-code-40f64b363663)

[26 GitHub Repositories To Inspire Your Next Data Science Project](https://towardsdatascience.com/26-github-repositories-to-inspire-your-next-data-science-project-3023c24f4c3c)

[7 Amazing Python Libraries For Natural Language Processing](https://towardsdatascience.com/7-amazing-python-libraries-for-natural-language-processing-50ca6f9f5f11)

[4 Amazing Python Libraries That You Should Try Right Now](https://towardsdatascience.com/4-amazing-python-libraries-that-you-should-try-right-now-872df6f1c93)


[Tools for Efficient Deep Learning](https://towardsdatascience.com/tools-for-efficient-deep-learning-c9585122ded0)

[3 Simple Ways to get started on NLP Sentiment Analysis](https://medium.com/geekculture/3-simple-ways-to-get-started-on-nlp-sentiment-analysis-d0d102ef5bf8)


