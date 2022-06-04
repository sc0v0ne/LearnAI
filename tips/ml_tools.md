# Machine Learning Tools

<!-- MarkdownTOC levels=1,2 -->

- How to choose an ML framework
- Keras Tutorials
- Data Exploration Tools
- ML Libraries
- Computer Vision
- Feature Engineering Tools
- Pretrained Model Libraries
- Python Libraries
- Plots and Graphs
- Deep Learning Tools
- Time Series
- Audio
- Browser Extensions for Web Developers
- Websites for Developers
- CSS Websites for Developers
- JetBrains Plugins
- Linux Utilities
- References

<!-- /MarkdownTOC -->

Here is a list of ML tools that I have found to be helpful for AI engineering.

For items without links, see **Github Lists**. 

[GitHub Lists](https://github.com/codecypher?tab=stars)

[Github Student Developer Pack](https://education.github.com/pack)


[HuggingFace Spaces](https://huggingface.co/spaces/launch)


## How to choose an ML framework

[Keras vs PyTorch for Deep Learning](https://towardsdatascience.com/keras-vs-pytorch-for-deep-learning-a013cb63870d)


## Keras Tutorials

[Introduction to Keras for Engineers](https://keras.io/getting_started/intro_to_keras_for_engineers/)

[3 ways to create a Machine Learning model with Keras and TensorFlow 2.0 (Sequential, Functional, and Model Subclassing)](https://towardsdatascience.com/3-ways-to-create-a-machine-learning-model-with-keras-and-tensorflow-2-0-de09323af4d3)

[Introducing TensorFlow Datasets](https://medium.com/tensorflow/introducing-tensorflow-datasets-c7f01f7e19f3)

[Getting started with TensorFlow 2.0](https://medium.com/@himanshurawlani/getting-started-with-tensorflow-2-0-faf5428febae)

[Introducing TensorFlow Datasets](https://medium.com/tensorflow/introducing-tensorflow-datasets-c7f01f7e19f3)

[How to (quickly) Build a Tensorflow Training Pipeline](https://towardsdatascience.com/how-to-quickly-build-a-tensorflow-training-pipeline-15e9ae4d78a0?gi=f2df1cc3455f)


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


## Feature Engineering Tools

There are many tools that will help you in automating the entire feature engineering process and producing a large pool of features in a short period of time for both classification and regression tasks.

- Feature-engine
- Featuretools
- AutoFeat

### Tutorials

[The Only Web Scraping Tool you need for Data Science](https://medium.com/nerd-for-tech/the-only-web-scraping-tool-you-need-for-data-science-f388e2afa187)



## Pretrained Model Libraries

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



## Plots and Graphs

[Plots](./plots.md)



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



## Time Series

- statsmodels
- stumpy
- AutoTS
- Darts
- TsFresh


## Audio

[AssemblyAI](https://www.assemblyai.com/)



## Browser Extensions for Web Developers

- Awesome Screenshots
- Colorzilla
- Form Vault
- Google Font Previewer
- JsonVue
- Octotree
- Site Pallette
- Temp Email
- User JavaScript and CSS
- Wappalyzer
- Web Developer Checklist
- WhatFont


## Websites for Developers

- [Browser frame](https://browserframe.com/)
- [Can I use](https://caniuse.com/?search=Grid)
- [Codepen](https://codepen.io/)
- [DevDocs](https://devdocs.io/)
- [LambdaTest](https://www.lambdatest.com/)
- [Meta Tags](https://metatags.io/)
- [Peppertype](https://www.peppertype.ai/)
- [Profile Pic Maker](https://pfpmaker.com/)
- [Regex101](https://regex101.com/)
- [Resume.io](https:/resume.io)
- [Roadmap](https://roadmap.sh/)
- [Small Dev Tools](https://smalldev.tools/)
- [TypeScript Playground](https://www.typescriptlang.org/)
- [Web Page Test](https://www.webpagetest.org/)


## CSS Websites for Developers

- [Animation pausing](https://waitanimate.wstone.uk/)
- [Color Palette Generator](https://mybrandnewlogo.com/color-palette-generator)
- [CSS Generator](https://html-css-js.com/css/generator/box-shadow/)
- [Cubic Bezier Generator](https://cubic-bezier.com/#.17,.67,.83,.67)
- [Gradient Generator](https://cssgradient.io/)
- [Grid Generator](https://cssgrid-generator.netlify.app/)
- [Hamburgers](https://jonsuh.com/hamburgers/)
- [Layout Generator](https://layout.bradwoods.io/)
- [Layout Patterns](https://web.dev/patterns/layout/)
- [Responsively](https://responsively.app/)
- [SchemeColor](https://www.schemecolor.com/)
- [SVG Generator](https://haikei.app/)
- [Transition Animation Generator](https://www.transition.style/)

### [Animista](https://animista.net/)

CSS Animation can get tedious to work with. 

By using Animista, we are able to work interactively with animations.

### [Clip Path Generator](https://bennettfeely.com/clippy/)

```css
  clip-path: polygon(25% 0%, 75% 0%, 100% 53%, 25% 100%, 0% 50%);
```

### [Responsive Font Calculator](https://websemantics.uk/tools/responsive-font-calculator/)

We can easily create a fluid Typography experience which has wider support and can be implemented with a few CSS lines. This experience is just created by using the viewport width, and or height to smoothly scale the root font size. 

We can avoid the jumps that are created by just using media queries.

This web tool will make it for you to fine-tune and design a fluid experience for your users.0

All we have to do is configure the options and you will get a CSS output that you can paste to your side.

```css
  :root {
    font-size: clamp(1rem, calc(1rem + ((1vw - 0.48rem) * 0.6944)), 1.5rem);
    min-height: 0vw;
  }
```

### [Type Scale](https://type-scale.com/)

The fonts are a key aspect of any website, so we have another useful web app to help with fonts. 

When designing a website is it important to see how the different font sizes play together. Using this web app, it is simple to create a consistent font scale.

We can choose between 8 different predetermined scales or build our custom one. We just have to define a growth factor and the tool takes care of the rest.

This will generate fonts using the rem unit it is also handy to see how different base size fonts will look. The default is 16px which matches any browser's default root font.

Once we have everything looking good, we can copy the generated CSS or view the results in a codepen instance. 



## JetBrains Plugins

### GitLive

[GitLive](https://plugins.jetbrains.com/plugin/11955-gitlive) makes it easier to see the changes that teammates are making.

The plugin works offline with any Git repository and just uses the data from your local clone. 

There is also an online mode that requires you to sign in with GitHub, GitLab, Bitbucket or Azure Dev Ops. Then you can see who else from your team is online, what issue and branch they are working on, and even take a peek at their uncommitted changes — all updated in real time instead of on push/pull.

### Code Time

[Code Time](https://plugins.jetbrains.com/plugin/10687-code-time) is an open source plugin for automatic programming metrics and time tracking. Its advanced features can provide you with detailed feedback on how productive you are at work (a big plus for a slick design!).

### Git Machete

Producing small PRs is definitely a good practice, but it’s easy to get lost in multiple branches and stacked PRs. Git Machete is a useful plugin that helps you keep track of all the branches, their relationship with each other and with the remote repository.

Git Machete also aims at the automation of git actions and makes rebase/push/pull hassle-free (just a click on the button!), especially in the situation where there are a lot of branches and PRs.

The plugin automatically discovers branch layout and creates a tree-shaped graph of branches (in case of any changes or inaccuracies you can also modify it manually in .git/machete text file). The graph provides useful information about the branches: sync to parent status, sync to remote status, and custom annotation. There is also an option to toggle the unique commits for branches.

After right-clicking on a chosen branch in the graph, you can perform git actions like rebase/push/pull on that branch without a need to switch from your current branch, which is very handy!

### Smart Search

The Smart Search plugin is handy when you need to google something when coding. 

The JetBrains IDEs have a built-in Search with Google action, but Smart Search gives you more useful options such as Stack Overflow, GitHub, or Google Translate.


### Stepsize

Stepsize is an example of a tool that can save you time spent on tracking and addressing potential problems. 

Stepsize is an issue tracker inside your editor for managing technical debt and maintenance issues.



## Linux Utilities

- awk
- fkill
- hstr
- httpie
- rich
- ripgrep
- screen
- tmux

### awk

awk is a pattern scanning and text processing language which is also considered a programming language specifically designed for processing text.

### [Lucidchart](https://www.lucidchart.com/pages/)

Lucidchart is a diagraming tool that also has shared space for collaboration and the ability to make notes next to diagrams.

### [Screen](https://linuxize.com/post/how-to-use-linux-screen/)

Screen is a GNU linux utility that lets you launch and use multiple shell sessions from a single ssh session. The process started with screen can be detached from session and then reattached at a later time. So your experiments can be run in the background, without the need to worry about session closing, or terminal crashing.

### httpie

HTTPie is a command-line HTTP client. Its goal is to make CLI interaction with web services as human-friendly as possible. HTTPie is designed for testing, debugging, and generally interacting with APIs & HTTP servers. 

The http and https commands allow for creating and sending arbitrary HTTP requests using a simple and natural syntax and provide formatted and colourized output.

### Multipass

[Multipass](https://multipass.run) is a VM platform (perhaps better than VirtualBox and VMWare).

Multipass can be used on Ubuntu, Windows 10, and macOS (including Apple M1) with exactly the same command set.

[Use Linux Virtual Machines with Multipass](https://medium.com/codex/use-linux-virtual-machines-with-multipass-4e2b620cc6)

### rich

rich makes it easy to add colour and style to terminal output. It can also render pretty tables, progress bars, markdown, syntax highlighted source code, tracebacks, and more — out of the box.

### tmux

tmux is a terminal multiplexer that allows you to access a tmux terminal using multiple virtual terminals.

tmux takes advantage of a client-server model which allows you to attach terminals to a tmux session which means: 

- You can run several terminals at once concurrently off a single tmux session without spawning any new terminal sessions.

- Sudden disconnects from a cloud server running tmux will not kill the processes running inside the tmux session.

tmux also includes a window-pane mentality which means you can run more than one terminal on a single screen.

### Tutorials

[How To Install And Use tmux On Ubuntu 12.10](https://www.digitalocean.com/community/tutorials/how-to-install-and-use-tmux-on-ubuntu-12-10--2)

[10 Practical Uses of AWK Command for Text Processing](https://betterprogramming.pub/10-practical-use-of-awk-command-in-linux-unix-26fbd92f1112)

[Display Rich Text In The Console Using rich](https://towardsdatascience.com/get-rich-using-python-af66176ece8f?source=linkShare-d5796c2c39d5-1641842633)




## References

[All Top Python Libraries for Data Science Explained](https://towardsdatascience.com/all-top-python-libraries-for-data-science-explained-with-code-40f64b363663)

[26 GitHub Repositories To Inspire Your Next Data Science Project](https://towardsdatascience.com/26-github-repositories-to-inspire-your-next-data-science-project-3023c24f4c3c)

[7 Amazing Python Libraries For Natural Language Processing](https://towardsdatascience.com/7-amazing-python-libraries-for-natural-language-processing-50ca6f9f5f11)

[4 Amazing Python Libraries That You Should Try Right Now](https://towardsdatascience.com/4-amazing-python-libraries-that-you-should-try-right-now-872df6f1c93)


[Tools for Efficient Deep Learning](https://towardsdatascience.com/tools-for-efficient-deep-learning-c9585122ded0)

[4 Time-Saving Linux Tools To Improve Your Workflow](https://betterprogramming.pub/4-time-saving-linux-tools-to-improve-your-workflow-54604d574d53)

[5 JetBrains Plugins To Boost Your Productivity](https://betterprogramming.pub/5-jetbrains-plugins-to-boost-your-productivity-ecf5461ad642)


