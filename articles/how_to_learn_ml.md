# How to Learn Machine Learning

## Tips from an AI Engineer

As a Discord AI forum user, I have seen and answered a lot of questions from users that are getting started with artificial intelligence (AI) and machine learning (ML), so I thought I would sure share notes and advice on learning ML.

As a professional AI and software engineer with 25+ years of experience, I have a lot of experience in designing and implementing a variety of software applications and technologies. Thus, I have used and implemented hundreds (perhaps thousands) of various software libraries, frameworks, and APIs.

I have a M.S. in Mathematics and MSCS in Aritifical Intelligence, so I am more than qualified to mentor and teach computer science and software engineering courses.

Since this is an AI/ML website, I will assume that most readers will have the following goals:

- You are interested in becoming an AI/ML engineer.
- You are interested in learning/following software engineering best practices [1][2].

Here are some beginner mistakes to avoid when learning ML [3][4]:

- Avoid asking "What is the best algorithm for X problem?"
- Avoid trying to code ML algorithms from scratch
- Avoid trying to do things manually
- Avoid using neural networks (only as a last resort)

## Should I start learning ML by coding an algorithm from scratch?

As my professor once said "any competent software engineer can implement any AI/ML algorithm".

It is important to know what algorithms are available for a given problem, how they work, and how to get the most out of them. However, this does not mean you need to hand-code the algorithms from scratch.

There are far too many ML algorithms for a single software developer to ever code and understand all the details needed to properly implement them (and more algorithms are constantly being developed). Just concentrating on the "more interesting" or latest algorithm is a recipe for disaster and a common beginner mistake.

In fact, there are many other concepts that are much more important than knowing how to implement an ML algorithm (see next section for details).

There are plenty of other topics that you should explore that are much more important than trying to understand the internals of a particular ML algorithm.
Once you have mastered using ML algorithms, you can go back later and try to understand how they work internally by coding them by hand (if needed).

## Tips on Learning ML

Here are some beginner concepts for learning ML:

- How to define an ML problem?
- How to select a dataset?
- How to perform data preparation?
- How to perform feature engineering?
- How to select an ML algorithm?
- How to choose a performance metric?
- How to train an ML model?
- How to present ML results?
- How to design and deploy an ML solution?
- How to debug ML models?

In graduate school, we are taught that the proper technique for learning ML is to take a black-box view of ML algorithms (top-down approach) since it is best not to waste time trying to hand-code algorithms (bottom-up approach), especially when you are just getting started.

In general, you will find that there is very little to learn by focusing on the internal implementation of an algorithm. It is more important to focus on learning how best to use and deploy it.

In software engineering, there are many well-known principles such as DRY, KISS, and Occam's Razor. Perhaps the most important principle is "don't reinvent the wheel". Nevertheless, it seems as though every beginner in machine learning proceeds to try to implement a neural network from scratch. In fact, an ML engineer needs to know the best use of ALL software libraries, APIs, and algorithms that are used in an application, not just ML libraries.
Occam's Razor: The simplest solution is always the best.

A professional AI engineer knows that the current trend is a data-centric rather than model-centric approach to machine learning. Therefore, a beginner would be wise to focus on learning data preparation, feature engineering, dimenstionality reduction, and other important techniques rather than focus on the implementation details of a particular ML model.

In addition, a shift in mindset is required to be effective at machine learning from trying to find the "best" model/approach to "good enough".

Automation of the full software development life cycle is a big part of modern software development, so it is doubtful that you will implement and deploy a hand-coded ML algorithm (don't reinvent the wheel). In fact, there are a multitude of reasons to avoid writing custom code when robust, well-proven libraries and frameworks are readily available.

In short, it is best to understand the big picture (top-down approach) of the ML engineering process before delving into the implementation details (bottom-up approach).

## Avoid Code Fixation

In my experience, I have tried to help beginners only to find that they have spent hundreds of hours coding from a tutorial (or worse a Tensorflow or PyTorch documentation example) using the wrong model and/or solving the wrong problem. Thus, they are often argumentative and unwilling to listen when told that they have made a mistake which I call code fixation.

You should first spend time finding several good tutorials/examples that solve a similar problem that have been written by someone with credentials. Then, you need to spend time understanding your dataset to make sure it is being modeled correctly. For example, just because your data has a time feature does not mean that the data is best modeled as a time series and/or using neural networks (a common beginner mistake). First, evaluate several or many simple models on your dataset using an AutoML tool such as [Orange](https://orangedatamining.com) to find the best performing model(s) to be used as a baseline for comparison. Keep in mind that your chosen model must perform better than any simpler baseline model.

## Tutorials and Books

Choosing an ML learning resource is primarliy a matter of personal preference. In geneal, I recommend that you refer to more than one resource (other than scikit-learn, tensorflow, and PyTorch documentation) when learning ML.

J. Brownlee has an excellent series of tutorials for learning many ML topics: [Getting Started Guides](https://machinelearningmastery.com/start-here/)

If you have an .edu email account you can get free access to oreilly.com which has some good books for beginners as well as advanced books on various AI/ML topics.

## Books and Articles for Beginners

P. D. Smith, Hands-On Artificial Intelligence for Beginners, Birmingham, U.K.: Packt Pub. Ltd., 2018.

J. Eckroth, Python Artificial Intelligence Projects for Beginners, Birmingham, U.K.: Packt Pub. Ltd., 2018.

C. Y. Laporte, "Not Teaching Software Engineering Standards to Future Software Engineers - Malpractice?", IEEE Computer, May 2021.

## Books for Developers

I have found some of the books in the _Pragmatic Bookshelf_ series (available on oreilly.com) to be helpful:

- A Common-Sense Guide to Data Structures and Algorithms
- Programming Machine Learning
- Practical Machine Learning for Computer Vision

Here are some books on oreilly.com that I have found helpful:

A. Artasanchez and P. Joshi, Artificial Intelligence with Python, 2nd ed. Birmingham, U.K.: Packt Pub. Ltd., 2020.

D. Rothman, Artificial Intelligence by Example, 2nd ed. Birmingham, U.K.: Packt Pub. Ltd., 2020.

S. Raschka and V. Mirjalili, Python Machine Learning, 3rd ed., Birmingham, U.K.: Packt Pub. Ltd., 2019.

Y. Liu, Python Machine Learning by Example, 3rd ed., Birmingham, U.K.: Packt Pub. Ltd., 2020.

A. R. Jha, Mastering PyTorch, Birmingham, U.K.: Packt Pub. Ltd., 2021.

J. Matthew, PyTorch Artificial Intelligence Fundamentals, Birmingham, U.K.: Packt Pub. Ltd., 2020.

Here are some ML textbooks that I have found helpful:

S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 3rd ed. Upper Saddle River, NJ: Prentice Hall, ISBN: 978–0–13–604259–4, 2010.

E. Alpaydin, Introduction to Machine Learning, 3rd ed., MIT Press, ISBN: 978–0262028189, 2014.

S. Raschka. and V. Mirjalili, Python Machine Learning, 2nd ed. Packt, ISBN: 978–1787125933, 2017.

S. Bird, E. Klein, and E. Loper, Natural Language Processing with Python - Analyzing Text with the Natural Language Toolkit. Available online at https://www.nltk.org/book/

## Resources on Advanced Concepts

P. Palanisamy, Hands-On Intelligent Agents with OpenAI Gym, Birmingham, U.K.: Packt Pub. Ltd., 2018.

K. NG, A. Padmanabhan, and M. R. Cole, Mobile Artificial Intelligence Projects, Birmingham, U.K.: Packt Pub. Ltd., 2019.

F. X. Govers, Artificial Intelligence for Robotics, Birmingham, U.K.: Packt Pub. Ltd., 2018.

[Serving ML Models in Production: Common Patterns](https://www.kdnuggets.com/2021/10/serving-ml-models-production-common-patterns.html)

V. Lakshmanan, S. Robinson, M. Munn, Machine Learning Design Patterns, Sebastopol, CA: O'Reilly Media, Inc., 2021.

H. Lin and B. Biggio, "Adversarial Machine Learning: Attacks From Laboratories to the Real World," IEEE Computer, May 2021.

P. Palanisamy, Hands-On Intelligent Agents with OpenAI Gym, Birmingham, U.K.: Packt Pub. Ltd., 2018.

Finally, a professional software engineer should also be able to navigate the [SWEBOK](https://www.computer.org/education/bodies-of-knowledge/software-engineering/v3) and learn the key software engineering concepts that it contains since IEEE is the de facto authority on software and hardware technologies.

My article was mainly motivated by helping many Discord AI forum users who are interested in a self-study approach. However, if you are going to spend the time to study AI/ML then you might as well invest in an online degree which will greatly increase your career opportunities (and a requirement for most all AI/ML engineer positions).


## References

[1] P. Bourque and R. E. Fairley, Guide to the Software Engineering Body of Knowledge, v. 3, IEEE, 2014.

[2] [IEEE Code of Ethics](https://medium.com/r/?url=https%3A%2F%2Fwww.ieee.org%2Fabout%2Fcorporate%2Fgovernance%2Fp7-8.html)

[3] [5 Mistakes Programmers Make when Starting in Machine Learning](https://medium.com/r/?url=https%3A%2F%2Fmachinelearningmastery.com%2Fmistakes-programmers-make-when-starting-in-machine-learning%2F)

[4] [Stop Coding Machine Learning Algorithms From Scratch](https://medium.com/r/?url=https%3A%2F%2Fmachinelearningmastery.com%2Fdont-implement-machine-learning-algorithms%2F)

[Do you need a Master's Degree in Data Science?](https://medium.com/r/?url=https%3A%2F%2Ftowardsdatascience.com%2Fdo-you-need-a-masters-degree-in-data-science-you-are-asking-the-wrong-question-8c83dec8bf1b%3Fsource%3Drss----7f60cf5620c9---4)


[10] S. Russell and P. Norvig, Artificial Intelligence: A Modern Approach, 3rd ed. Upper Saddle River, NJ, USA: Prentice Hall, 2010.


[Applied Machine Learning Process](https://machinelearningmastery.com/start-here/#process)

[Understand Machine Learning Algorithms](https://machinelearningmastery.com/start-here/#algorithms)

