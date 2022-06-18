# Concurrency and Parallelism

<!-- MarkdownTOC -->

- Concurrency vs Parallelism
- Concurrency and Parallelism in Python
- Types of Parallelization
- References

<!-- /MarkdownTOC -->

**Concurrency** is about dealing with lots of things at once. 

**Parallelism** is about _doing_ lots of things at once. 

Concurrency is about structure whereas parallelism is about execution.  

Concurrency provides a way to structure a solution to solve a problem that may (but not necessarily) be parallelizable.


## Concurrency vs Parallelism

The goal of concurrency is to prevent tasks from blocking each other by switching among them whenever one is forced to wait on an external resource. 

Example: handling multiple network requests.

The better way would be to launch every request simultaneously and switch among them as we receive the responses. Thus, we eliminate the time spent waiting for the responses.


Parallelism is maximizing the use of resources by launching processes or threads that make use of all the CPU cores of the computer.

For parallelization, we would split the work among all the workers, so that the work will be done faster and each worker will do less work.


- Concurrency is best for tasks that depend on external resources such as I/O (shared resources).

- Parallelism is best for CPU-intensive tasks.


## Concurrency and Parallelism in Python

Python provides us with mechanisms to implement concurrency and parallelism

For concurrency, we have _multithreading_ and _async_ 

For parallelism, we have _multiprocessing_. 



## Types of Parallelization

Parallelization is possible in two ways:

- Multithreading: Using multiple threads of a process/worker

- Multiprocessing: Using multiple processors

Multithreading is useful for I/O bound applications such as when we have to download and upload multiple files.

Multiprocessing is useful for CPU-bound applications.


Here is anexample of one use case for multiprocessing.

Suppose we have 1000 images saved in a folder and for each image we need to perform the following operations:

- Convert image to grayscale
- Resize the grayscale image to a given size
- Save the modified image in a folder

Doing this process on each image is independent of each other -- processing one image would not affect any other image in the folder. 

Therefore, multiprocessing can help us reduce the total time. 

The total time will be reduced by a factor equal to the number of processors we use in parallel. This is one of many examples where you can use parallelization to save time.



## References

R. H. Arpaci-Dusseau and A. C. Arpaci-Dusseau, Operating Systems: Three Easy Pieces, 2018, v. 1.01, Available online: https://pages.cs.wisc.edu/~remzi/OSTEP/


[Concurrency and Parallelism: What is the difference?](https://towardsdatascience.com/concurrency-and-parallelism-what-is-the-difference-bdf01069b081)

[Parallelize your python code to save time on data processing](https://towardsdatascience.com/parallelize-your-python-code-to-save-time-on-data-processing-805934b826e2)

