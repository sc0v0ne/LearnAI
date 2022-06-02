# Performance

<!-- MarkdownTOC -->

- Why Python is so slow
    - Single-thread vs multi-threaded
    - How to speed things up
- Lightning Fast Iteration
    - Zip
    - Itertools
    - Stop Nesting
    - Do not Zip dicts!
    - Filter
- Optimize Python Code
    - Use built-in functions rather than coding them from scratch
    - Focus on Memory Consumption During Code Execution
    - Memoization in Python
    - Using C libraries/PyPy to Get Performance Gain
    - Proper Use of Data Structures and Algorithms
    - Avoid using + for string concatenation
    - Use tuple packing notation for swapping two variables
    - Use list comprehensions rather than loops to construct lists
    - Use chained comparisons
    - Use the in operator to test membership
    - Avoid global variables
    - Use enumerate if you need a loop index
    - Use the latest release of Python
- Optimize Memory Usage
    - Cache it
    - Sort big in place
    - Garbage collector
- Improve Python Performance
- Intel oneAPI AI Analytics Toolkit
    - Intel optimized Modin
    - Intel optimized Scikit-learn
    - Intel optimized XGBoost
    - Intel optimized TensorFlow and Pytorch
    - Intel optimized Python
    - Model Zoo for Intel Architecture
    - Intel Neural Compressor
- Speedup TensorFlow Training
    - Use @tf.function decorator
    - Optimize dataset loading
    - Use mixed precision
    - Accelerated Linear Algebra \(XLA\)
- Improve Tensorflow Performance
    - Mixed Precision on NVIDIA GPUs
    - Mix Precision in Tensorflow
    - Fusing multiple ops into one
    - Fusion with Tensorflow 2.x
- Keras GPU Performance
- Make It Easier to Work with Large Datasets
    - Explicitly pass the data-types
    - Select subset of columns
    - Convert dataframe to parquet
    - Convert to pkl
    - Dask
    - Modin
    - Vaex
    - Read using Pandas in Chunks
- Python Performance
- Scikit-learn Performance
- Tensorflow Performance
- Tensorflow GPU
- Tensorflow on macOs
- References

<!-- /MarkdownTOC -->


Here are some resources to improve Python performance, also see [Memory Usage](,/memory_usage.md)


## Why Python is so slow

Slowness vs waiting

1. CPU-tasks
2. I/O-tasks


Examples of I/O tasks are writing a file, requesting some data from an API, printing a page; they involve waiting. 

Although I/O can cause a program to take more time to execute, this is not Python’s fault. Python is just waiting for a response; a faster language cannot wait faster. 

Thus, I/O slowness is not what we are trying to solve here. 

Here, we figure out why Python executes CPU-tasks more slowly than other languages.

- Compiled vs Interpreted
- Garbage collection and memory management
- Single-thread vs multi-threaded


### Single-thread vs multi-threaded

Python is single-threaded on a single CPU by design. 

The mechanism that makes sure of this is called the GIL: the Global Interpreter Lock. The GIL makes sure that the interpreter executes only one thread at any given time.

The problem the GIL solves is the way Python uses reference counting for memory management. A variable’s reference count needs to be protected from situations where two threads simultaneously increase or decrease the count which can cause all kinds of weird bugs to to memory leaks (when an object is no longer necessary but is not removed) or incorrect release of the memory (a variable gets removed from the memory while other variables still need it). 

In short: Because of the way garbage collection is designed, Python has to implements a GIL to ensure it runs on a single thread. There are ways to circumvent the GIL though, read this article, to thread or multiprocess your code and speed it up significanly.

### How to speed things up

Thus, we can conclude that the main problems for execution speed are:

- **Interpretation:** compilation and interpretation occurs during runtime due to the dynamic typing of variables. For the same reason we have to create a new PyObject, pick an address in memory and allocate enough memory every time we create or “overwrite” a “variable” we create a new PyObject for which memory is allocated.

- **Single thread:** The way garbage-collection is designed forces a GIL: limiting all executing to a single thread on a single CPU

How do we remedy these problems:

- Use built-in C-modules in Python such as range(). 

- I/O-tasks release the GIL so they can be threaded; you can wait for many tasks to finish simultaneously. 

- Run CPU-tasks in parallel by multiprocessing. 

- Create and import your own C-module into Python; you can extend Python with pieces of compiled C-code that are 100x faster than Python.

- Write Python-like code that Cython compiles to C and then neatly packages into a Python package which offers the readability and easy syntax of Python with the speed of C. 

----------


## Lightning Fast Iteration

Here are some tips to improve Python loop/iteration performance [2].

### Zip

```py
    z = list([1, 2, 3, 4, 5, 6])
    z2 = list([1, 2, 3, 4, 5, 6])

    # create a new zip iterator class
    bothlsts = zip(z, z2)

    for i, c in bothlsts:
        print(i + c)

    # call the zip() class directly
    for i, c in zip(z, z2):
        print(i + c)
```

### Itertools

```py
    import itertools as its
    
    def fizz_buzz(n):
        fizzes = its.cycle([""] * 2 + ["Fizz"])
        buzzes = its.cycle([""] * 4 + ["Buzz"])
        fizzes_buzzes = (fizz + buzz for fizz, buzz in zip(fizzes, buzzes))
        result = (word or n for word, n in zip(fizzes_buzzes, its.count(1)))
    
        for i in its.islice(result, 100):
            print(i)
```


### Stop Nesting

Avoid writing nested for loops.

If you need an index to call, you can use the `enumerate()` on your iterator in a similar fashion to how we used `zip()` above.

### Do not Zip dicts!

There is no need to use zip() with dictionaries.

```py
    dct = {"A" : [5, 6, 7, 8], "B" : [5, 6, 7, 9]}

    for i in dct:
        print(i)
    # A, B

    for i in dct:
        print(dct[i])
    # [5, 6, 7, 8]

    # only work with the values
    for i in dct.values():
        print(i)
```

### Filter

The built-in Python `filter()` method can be used to eliminate portions of an iterable with minimal performance cost.

```py
    people = [{"name": "John", "id": 1}, {"name": "Mike", "id": 4}, 
              {"name": "Sandra", "id": 2}, {"name": "Jennifer", "id": 3}]

    # filter out some of the unwanted values prior to looping
    for person in filter(lambda i: i["id"] % 2 == 0, people):
        print(person)

    # {'name': 'Mike', 'id': 4}
    # {'name': 'Sandra', 'id': 2}
```




## Optimize Python Code

Tips to make python code more optimized and improve performance [2].

### Use built-in functions rather than coding them from scratch

Some built-in functions in Python like map(), sum(), max(), etc. are implemented in C so they are not interpreted during the execution which saves a lot of time.

For example, if you want to convert a string into a list you can do that using the `map()` function  instead of appending the contents of the strings into a list manually.

```py
    string = ‘Australia’
    U = map(str, s)
    print(list(string))
    # [‘A’, ‘u’, ‘s’, ‘t’, ‘r’, ‘a’, ‘l’, ‘i’, ‘a’]
```

Also, the use of f-strings while printing variables in a string instead of the traditional ‘+’ operator is also very useful in this case.


### Focus on Memory Consumption During Code Execution

Reducing the memory footprint in your code definitely make your code more optimized. 

Check if unwanted memory consumption is occuring. 

Example: str concatenation using + operator will generate a new string each time which will cause unwanted memory consumption. Instead of using this method to concatenate strings, we can use the function `join()` after taking all the strings in a list.

### Memoization in Python

Those who know the concept of dynamic programming are well versed with the concept of memorization. 

In memorization, the repetitive calculation is avoided by storing the values of the functions in the memory. 

Although more memory is used, the performance gain is significant. Python comes with a library called `functools` that has an LRU cache decorator that can give you access to a cache memory that can be used to store certain values.

### Using C libraries/PyPy to Get Performance Gain

If there is a C library that can do your job then it’s better to use that to save time when the code is interpreted. 

The best way to do that is to use the ctype library in python. 

There is another library called CFFI which provides an elegant interface to C.

If you do not want to use C then you could use the PyPy package due to the presence of the JIT (Just In Time) compiler which gives a significant boost to your Python code.

### Proper Use of Data Structures and Algorithms

This is more of a general tip but it is the most important one as it can give you a considerable amount of performance boost by improving the time complexity of the code.

For example, It is always a good idea to use dictionaries instead of lists in python in case you don’t have any repeated elements and you are going to access the elements multiple times.

This is because the dictionaries use hash tables to store the elements which have a time complexity of O(1) when it comes to searching compared to O(n) for lists in the worst case. So it will give you a considerable performance gain.

### Avoid using + for string concatenation

```py
    s = ", ".join((a, b, c))
```

### Use tuple packing notation for swapping two variables

```py
    a, b = b, a
```

### Use list comprehensions rather than loops to construct lists

```py
    b = [x*2 for x in a]
```

### Use chained comparisons

If you need to compare a value against an upper and lower bound, you can (and should) used operator chaining:

```py
    if 10 < a < 100:
        x = 2 * x
    
    if 10 < f(x) < 100:
        x = f(x) + 10
```

### Use the in operator to test membership

If you want to check if a particular value is present in a list, tuple, or set, you should use the in operator:

```py
    k = [1, 2, 3]
    if 2 in k:
        # ...
```

### Avoid global variables

A global variable is a variable that is declared at the top level so that it can be accessed by any part of the program.

While it can be very convenient to be able to access shared data from anywhere, it usually causes more problems than it solves, mainly because it allows any part of the code to introduce unexpected side effects. So globals are generally to be avoided. But if you need an extra reason to not use them, they are also slower to access.

### Use enumerate if you need a loop index

If for some reason you really need a loop index, you should use the enumerate function which is faster and clearer:

```py
    for i, x in enumerate(k):
        print(i, x)
```

### Use the latest release of Python

New versions of Python are released quite frequently (at the time of writing Python 3.9 has been updated 8 times in the last year). It is worth keeping up to date as new versions often have bug fixes and security fixes, but they sometimes have performance improvements too.


## Optimize Memory Usage

Here are some tips to optimize memory usage in python [3].

### Cache it

In general, we want to cache anything that we download unless we know for certian that we will not  need it again or it will expire before we need it again.

- A classical approach to caching is to organize a directory for storing the previously obtained objects by their identifiers. The identifiers may be, for example, objects’ URLs, tweet ids, or database row numbers; anything related to the objects’ sources.

- The next step is to convert an identifier to a uniform-looking unique file name. We can write the conversion function ourselves or use the standard library. Start by encoding the identifier, which is presumably a string. 

Apply one of the hashing functions such as the hashlib.md5() or hashlib.sha256() which is faster to get a HASH object. 

The functions do not produce totally unique file names but the likelihood of getting two identical file names (called a hash collision) is so low that we can ignore it for all practical purposes. 

- Obtain a hexadecimal digest of the object which is a 64-character ASCII string: a perfect file name that has no resemblance to the original object identifier.

Assuming that the directory cache has already been created and is writable, we can pickle our objects into it.

```py
    import hashlib

    source = "https://lj-dev.livejournal.com/653177.html"
    hash = hashlib.sha256(source.encode())
    filename = hash.hexdigest()
    print(hash, filename)

    # First, check if the object has already been pickled.
    cache = f'cache/{filename}.p' 
    try:
      with open(cache, 'rb') as infile:
          # Has been pickled before! Simply unpickle 
          object = pickle.load(infile)
    except FileNotFoundError:
        # Download and pickle
        object = 'https://lj-dev.livejournal.com/653177.html' 
        with open(cache, 'wb') as outfile:
          pickle.dump(outfile, object) 
    except:
        # Things happen...
```

### Sort big in place

Sorting and searching are arguably the two most frequent and important operations in modern computing. 

Sorting and searching are so important that Python has two functions for sorting lists: list.sort() and sorted().

- `sorted()` sorts any iterable while `list.sort()` sorts only lists.
- `sorted()` creates a sorted copy of the original iterable.
- The `list.sort()` method sorts the list in place. 

The `list.sort()` method  shuffles the list items around without making a copy. If we could load the list into memory, we could surely afford to sort it. However, list.sort() ruins the original order. 

In summary, if our list is large then sort it in place with `list.sort()`. If our list is moderately sized or needs to preserve the original order, we can call `sorted()` and retrieve a sorted copy.

### Garbage collector

Python is a language with implicit memory management. The C and C++ languages require that we allocate and deallocate memory ourselves, but Python manages allocation and deallocation itself. 

When we define a variable through the assignment statement, Python creates the variable and the objects associated with it.

Each Python object has a reference count, which is the number of variables and other objects that refer to this object. When we create an object and do not assign it to a variable, the object has zero references.

When we redefine a variable, it no longer points to the old object and the reference count decreases.

```py
    'Hello, world!'         # An object without references
    s3 = s2 = s1 = s        # Four references to the same object!
    s = 'Goodbye, world!'   # Only three references remain
    strList = [s1]
    s1 = s2 = s3 = None     # Still one reference
```

When the reference count becomes zero, an object becomes unreachable. For most practical purposes, an unreachable object is a piece of garbage. A part of Python runtime called garbage collector automatically collects and discards unreferenced objects. There is rarely a need to mess with garbage collection, but here is a scenario where such interference is helpful.

Suppose we work with big data — something large enough to stress our computer’s RAM. 

We start with the original data set and progressively apply expensive transformations to it and record the intermediate results. An intermediate result may be used in more than one subsequent transformation. Eventually, our computer memory will be clogged with large objects, some of which are still needed while some aren’t. 

We can help Python by explicitly marking variables and objects associated with them for deletion using the `del` operator.

```py
    bigData = ...
    bigData1 = func1(bigData) 
    bigData2 = func2(bigData)
    del bigData # Not needed anymore
```

Bear in mind that del doesn’t remove the object from memory. It merely marks it as unreferenced and destroys its identifier. The garbage collector still must intervene and collect the garbage. 

We may want to force garbage collection immediately in anticipation of heavy memory use.

```py
    import gc # Garbage Collector
    gc.collect()
```

NOTE: Don not abuse this feature. Garbage collection takes a long time, so we should only let it happen only when necessary.



----------


## Improve Python Performance

The rule of thumb is that a computer is a sum of its parts or weakest link. In addition, the basic performance equation reminds us that there are always tradeoffs in hardware/software performance. 

Thus, there is no silver bullet hardware or software technology that will magically improve computer performance.

Hardware upgrades (vertical scaling) usually provide only marginal improvement in performance. However, we can achieve as much as 30-100x performance improvement using software libraries and code refactoring to improve parallelization (horizontal scaling) [2][3].

> There is no silver bullet to improve performance.

In general, improving computer performance is a cumulative process of several or many different approaches primarily software related.

> NOTE: Many of the software libraries to improve pandas performance also enhance numpy performance as well.



----------


## Intel oneAPI AI Analytics Toolkit

There are different hardware architectures such as CPU, GPU, FPGAs, AI accelerators, etc. The code written for one architecture cannot easily run on another architecture. 

> Intel oneAPI is mainly for Linux64

Therefore, Intel ceeated a unified programming model called oneAPI to solve this very same problem. With oneAPI, it does not matter which hardware architectures (CPU, GPU, FGPA, or accelerators) or libraries or languages, or frameworks you use, the same code runs on all hardware architectures without any changes and additionally provides performance benefits.

```bash
  conda install -c intel intel-aikit
```

### Intel optimized Modin

With Intel optimized Modin, you can expect further speed improvement on Pandas operations. 

```bash
  conda create -n aikit-modin -c intel intel-aikit-modin  # install in new environment
  conda install -c intel intel-aikit-modin
```

> intel-aikit-modin package includes Intel Distribution of Modin, Intel Extension for Scikit-learn, and Intel optimizations for XGboost. 


### Intel optimized Scikit-learn

The Intel optimized Scikit-learn helps to speed the model building and inference on single and multi-node systems.

```bash
  conda install -c conda-forge scikit-learn-intelex
```

```py
  from sklearnex import patch_sklearn
  patch_sklearn()
  
  # undo the patching
  sklearnex.unpatch_sklearn()
```

### Intel optimized XGBoost

The XGBoost is one of the most widely used boosting algorithms in data science. In collaboration with the XGBoost community, Intel has optimized the XGBoost algorithm to provide high-performance w.r.t. model training and faster inference on Intel architectures.

```py
  import xgboost as xgb
```

### Intel optimized TensorFlow and Pytorch

In collaboration with Google and Meta (Facebook), Intel has optimized the two popular deep learning libraries TensorFlow and Pytorch for Intel architectures. By using Intel-optimized TensorFlow and Pytorch, you will benefit from faster training time and inference.

To use Intel-optimized TensorFlow and Pytorch, you do not have to modify anything. You just need to install `intel-aikit-tensorflow` or `intel-aikit-pytorch`. 

```bash
  conda create -n aikit-tf -c intel intel-aikit-tensorflow  # default is tensorflow 2.5
  conda create -n aikit-pt -c intel intel-aikit-pytorch
  
  # workaround unable to resolve environment for intel-aikit-tensorflow
  conda create -n aikit-tf

  # install tensorflow v 2.6
  # incompatible with intel-aikit-modin=2021.4.1
  mamba install -c intel intel-aikit-tensorflow=2022.0.0  # tensorflow v 2.6

  # needed after install intel-aikit-tensorflow
  mamba install python-flatbuffers
```

### Intel optimized Python

The AI Analytics Toolkit also comes with Intel-optimized Python. 

When you install any of the above-mentioned tools (Modin, TensorFlow, or Pytorch), Intel optimized Python is also installed by default.

This Intel distribution of Python includes commonly used libraries such as Numpy, SciPy, Numba, Pandas, and Data Parallel Python. 

All these libraries are optimized to provide high performance which is achieved with the efficient use of multi-threading, vectorization, and more importantly memory management.


### Model Zoo for Intel Architecture

[Intel Model Zoo](https://github.com/IntelAI/models) contains links to pre-trained models (such as ResNet, UNet, BERT, etc.), sample scripts, best practices, and step-by-step tutorials to run popular machine learning models on Intel architecture.


### Intel Neural Compressor

[Intel Neural Compressor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/neural-compressor.html) is an open-source Python library that helps developers to deploy low-precision inference solutions on popular deep learning frameworks — TensorFlow, Pytorch, and ONNX.

The tool automatically optimizes low-precision recipes by applying different compression techniques such as quantization, pruning, mix-precision, etc. and thereby increasing inference performance without losing accuracy.

----------

[Installing Intel Distribution for Python and Intel Performance Libraries with Anaconda](https://www.intel.com/content/www/us/en/developer/articles/technical/using-intel-distribution-for-python-with-anaconda.html)

[Intel Optimization for TensorFlow Installation Guide](https://www.intel.com/content/www/us/en/developer/articles/guide/optimization-for-tensorflow-installation-guide.html)

[How to Speed up Scikit-Learn Model Training](https://medium.com/distributed-computing-with-ray/how-to-speed-up-scikit-learn-model-training-aaf17e2d1e1)


```py
  # Install intel conda packages with Continuum's Python (version conflicts on Linux)
  conda install mkl intel::mkl --no-update-deps
  conda install numpy intel::numpy --no-update-deps

  # macOS: AttributeError: module 'numpy' has no attribute 'ndarray'

  # Needed on macOS
  conda install -c intel numpy=1.19.5 --no-update-deps

  # Install intel optimization for tensorflow from anaconda channel 
  # cannot install tensorflow-mkl on macOS (version conflicts)
  conda install -c anaconda tensorflow
  conda install -c anaconda tensorflow-mkl

  # Install intel optimization for tensorflow from intel channel
  # conda install tensorflow -c intel
```

```py
  # Intel Extension for Scikit-learn
  conda install -c conda-forge scikit-learn-intelex

  from sklearnex import patch_sklearn
  patch_sklearn()
```


## Speedup TensorFlow Training

### Use @tf.function decorator

In TensorFlow 2, there are two execution modes: eager execution and graph execution. 

  1. In eager execution mode, the user interface is more intuitive but it suffers from performance issues because every function is called in a Python-native way. 

  2. In graph execution mode, the user interface is less intuitive but offers better performance due to using Python-independent dataflow graphs.

The `@tf.function` decorator allows you to execute functions in graph execution mode. 

```py
@tf.function
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b) 
 
def dense_layer(x, w, b):
    return add(tf.matmul(x, w), b)
```

When the @tf.function-decorated function is first called, a “tracing” takes place and a tf.Graph is generated. Once the graph is generated, the function will be executed in graph execution mode with no more tracing.

If you call the function with Python-native values or tensors that continuously change shapes, a graph will be generated every time which causes a lack of memory and slows down training.

### Optimize dataset loading

1. Use TFRecord format to save and load your data

Using the TFRecord format, we can save and load data in binary form that makes encoding and decoding the data much faster. 

2. Utilize tf.data.Dataset methods to efficiently load your data

TensorFlow2 supports various options for data loading.

2.1 num_parallel_calls option in interleaves and map method

The `num_parallel_calls` option represents a number of elements processed in parallel. 

We can set this option to `tf.data.AUTOTUNE` to dynamically adjust the value based on available hardware.

```py
dataset = Dataset.range(1, 6)
dataset = dataset.map(lambda x: x + 1, num_parallel_calls=tf.data.AUTOTUNE)
```

2.2 use prefetch method

The `prefetch` method creates a Dataset which prepares the next element while the current one is being processed. 

Most dataset pipelines are recommended to be ended with the `prefetch` method.

```py
dataset = tf.data.Dataset.range(3)
dataset = dataset.prefetch(2)
```

### Use mixed precision

Mixed precision is the use of both 16-bit and 32-bit floating-point types in a model during training to make it run faster and use less memory. 

In fact, forward/backward-propagation is computed in float16 and gradients are scaled to a proper range afterward. 

Since NVIDIA GPU generally runs faster in float16 rather than in float32, this reduces time and memory costs. 

Here are the steps to enable mixed precision for training. 

1. Set the data type policy

Add a line at the very beginning of the training code to set a global data type policy.

```py
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

2. Fix the data type of the output layer to float32

Add a line at the end of your model code or pass dtype='float32' to the output layer of your model.

```py
outputs = layers.Activation('linear', dtype='float32')(outputs)
```

3. Wrap your optimizer with LossScaleOptimizer

Wrap your optimizer with `LossScaleOptimizer` for loss scaling if you create a custom training loop instead of using `keras.model.fit`.

```py
optimizer = keras.optimizers.RMSprop()
optimizer = mixed_precision.LossScaleOptimizer(optimizer)
```

### Accelerated Linear Algebra (XLA)

Accelerated Linear Algebra (XLA) is a domain-specific compiler for linear algebra that can accelerate TensorFlow models without almost no source code changes.

XLA compiles the TensorFlow graph into a sequence of computation kernels generated specifically for the given model. 

Without XLA, TensorFlow graph executes three kernels: one for addition, one for multiplication, and one for reduction. However, XLA compiles these three kernels into one kernel so that intermediate results no longer have to be saved during the computation.

Using XLA, we can use less memory and also speed up training.

Enabling XLA is as simple as using the `@tf.function` decorator. 

```py
  # enable XlA globally
  tf.config.optimizer.set_jit(True)
  
  @tf.function(jit_compile=True)
  def dense_layer(x, w, b):
      return add(tf.matmul(x, w), b)
 ```
     

 
## Improve Tensorflow Performance

### Mixed Precision on NVIDIA GPUs

Mixed precision (MP) training offers significant computational speedup by performing operations in the half-precision format while storing minimal information in single-precision to retain as much information as possible in critical parts of the network.

MP uses both 16bit and 32bit floating point values to represent variables to reduce the memory requirements and to speed up training. MP relies on the fact that modern hardware accelerators such as GPUs and TPUs can run computations faster in 16bit.

There are numerous benefits to using numerical formats with lower precision than 32-bit floating-point: require less memory; require less memory bandwidth. 

- Speeds up math-intensive operations such as linear and convolution layers by using Tensor Cores.

- Speeds up memory-limited operations by accessing half the bytes compared to single-precision.

- Reduces memory requirements for training models, enabling larger models or larger mini-batches.

Among NVIDIA GPUs, those with compute capability 7.0 or higher will see the greatest performance benefit from mixed-precision because they have special hardware units called Tensor Cores to accelerate float16 matrix multiplications and convolutions.


### Mix Precision in Tensorflow

The mixed precision API is available in TensorFlow 2.1 with Keras interface. 

To use mixed precision in Keras, we have to create a _dtype policy_ which specify the dtypes layers will run in. 

Then, layers created will use mixed precision with a mix of float16 and float32.

```py
  from tensorflow.keras.mixed_precision import experimental as mixed_precision
  
  policy = mixed_precision.Policy('mixed_float16')
  mixed_precision.set_policy(policy)
  
  # Now design your model and train it
```

```py
import tensorflow as tf

try:
  tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
except ValueError:
  tpu = None

if tpu:
  policyConfig = 'mixed_bfloat16'
else: 
  policyConfig = 'mixed_float16'
policy = tf.keras.mixed_precision.Policy(policyConfig)
tf.keras.mixed_precision.set_global_policy(policy)
view raw mixed
```

> NOTE: Tensor Cores provide mix precision which requires certain dimensions of tensors such as dimensions of your dense layer, number of filters in Conv layers, number of units in RNN layer to be a multiple of 8.

To compare the performance of mixed-precision with float32, change the policy from `mixed_float16` to float32 which can improve performance up to 3x.

For numerical stability it is recommended that the model’s output layers use float32. This is achieved by either setting dtype=tf.float32 in the last layer or activation, or by adding a linear activation layer tf.keras.layers.Activation("linear", dtype=tf.float32) right at the model’s output. 

In addition data must be in float32 when plotting model predictions with matplotlib since plotting float16 data is not supported.

If you train your model with `tf.keras.model.fit` then you are done! If you implement a custom training loop with mixed_float16 a further step is required called _loss scaling_.

Mixed precission can speed up training on certain GPUs and TPUs. 

When using `tf.keras.model.fit`  to train your model, the only step required is building the model with mixed precission by using a global policy. 

If a custom training loop is implemented, the optimizer wrapper `tf.keras.mixed_precission.LossScaleOptimizer` should be implemented to prevent overflow and underflow.

### Fusing multiple ops into one

Usually when you run a TensorFlow graph, all operations are executed individually by the TensorFlow graph executor which means each op has a pre-compiled GPU kernel implementation. 

Fused Ops combine operations into a single kernel for improved performance.

Without fusion, without XLA, the graph launches three kernels: one for the multiplication, one for the addition and one for the reduction.

```py
  def model_fn(x, y, z): 
      return tf.reduce_sum(x + y * z)
```

Using op fusion, we can compute the result in a single kernel launch by fusing the addition, multiplication, and reduction into a single GPU kernel.

### Fusion with Tensorflow 2.x

Newer Tensorflow versions come with XLA which does fusion along with other optimizations for us.

Fusing ops together provides several performance advantages:

- Completely eliminates Op scheduling overhead (big win for cheap ops)

- Increases opportunities for ILP, vectorization etc.

- Improves temporal and spatial locality of data access



## Keras GPU Performance

```py
    import os
    
    from tensorflow.python.client import device_lib
    
    # disable oneDNN warning
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

    print(f"tensorflow version is {tf.__version__}")

    # Check if tensorflow is using GPU
    print(f"\nPhysical Devices:\n{tf.config.list_physical_devices('GPU')}")
    print(f"\n\nLocal Devices:\n{device_lib.list_local_devices()}")
    print(f"\nNum GPUs Available: {len(tf.config.list_physical_devices('GPU'))}")
```

```py
    import tensorflow as tf
    import tensorflow_datasets as tfds

    from tensorflow.python.framework.ops import disable_eager_execution
    from tensorflow.python.compiler.mlcompute import mlcompute

    #disable_eager_execution()

    mlcompute.set_mlc_device(device_name='gpu')
```


----------


## Make It Easier to Work with Large Datasets

Pandas mainly uses a single core of CPU to process instructions and does not take advantage of scaling up the computation across various cores of the CPU to speed up the workflow [8]. 

Thus, Pandas can cause memory issues when reading large datasets since it fails to load larger-than-memory data into RAM.

There are various other Python libraries that do not load the large data at once but interacts with system OS to map the data with Python. In addition, they utilize all the cores of the CPU to speed up the computations. 

The article [8] provides some tips on working with  huge datasets using pandas:

- Explicitly pass the data-types
- Select subset of columns
- Convert dataframe to parquet
- Convert to pkl

### Explicitly pass the data-types

```py
    import pandas as pd
    df = pd.read_csv(data_file, 
                     n_rows = 100, 
                     dtype={'col1': 'object', 'col2': 'float32',})
```

### Select subset of columns

```py
    cols_to_use = ['col1', 'col2',]
    df = pd.read_csv(data_file, usecols=cols_to_use)
```

### Convert dataframe to parquet

```py
    df.to_parquet()
    df = pd.read_parquet()
```

### Convert to pkl

```py
    df.to_pickle(‘train.pkl’) 
```

The article [7] discusses four Python libraries that can read and process large-sized datasets.

- Dask
- Modin
- Vaex
- Pandas with chunks

### Dask

Dask is an open-source Python library that provides multi-core and distributed parallel execution of larger-than-memory datasets

Dask provides the high-performance implementation of the function that parallelizes the implementation across all the cores of the CPU.

Dask provides API similar to Pandas and Numpycwhich makes it easy for developers to switch between the libraries.

```py
    import dask.dataframe as dd

    # Read the data using dask
    df_dask = dd.read_csv("DATA/text_dataset.csv")

    # Parallelize the text processing with dask
    df_dask['review'] = df_dask.review.map_partitions(preprocess_text)
```

### Modin

Modin is another Python library that speeds up Pandas notebooks, scripts, or workflows. 

Modin distributes both data and computations. 

Modin partitions a DataFrame along both axes so it performs on a matrix of partitions.

In contrast to Pandas, Modin utilizes all the cores available in the system, to speed up the Pandas workflow, only requiring users to change a single line of code in their notebooks.

```py
    import modin.pandas as md

    # read data using modin
    modin_df = pd.read_csv("DATA/text_dataset.csv")

    # Parallel text processing of review feature 
    modin_df['review'] = modin_df.review.apply(preprocess_text)
```

### Vaex

Vaex is a Python library that uses an _expression system_ and _memory mapping_ to interact with the CPU and parallelize the computations across various cores of the CPU.

Instead of loading the entire data into memory, Vaex just memory maps the data and creates an expression system.

Vaex covers some of the API of pandas and is efficient to perform data exploration and visualization for a large dataset on a standard machine.

```py
    import vaex

    # Read the data using Vaex
    df_vaex = vaex.read_csv("DATA/text_dataset.csv")

    # Parallize the text processing
    df_vaex['review'] = df_vaex.review.apply(preprocess_text)
```


### Read using Pandas in Chunks

Pandas loads the entire dataset into RAM which may cause a memory overflow issue while reading large datasets.

Instead, we can read the large dataset in _chunks_ and perform data processing for each chunk.

The idea is to load 10k instances in each chunk (lines 11–14), perform text processing for each chunk (lines 15–16), and append the processed data to the existing CSV file (lines 18–21).

```py
    # append to existing CSV file or save to new file
    def saveDataFrame(data_temp):
        
        path = "DATA/text_dataset.csv"
        if os.path.isfile(path):
            with open(path, 'a') as f:
                data_temp.to_csv(f, header=False)
        else:
            data_temp.to_csv(path, index=False)
            
    # Define chunksize
    chunk_size = 10**3

    # Read and process the dataset in chunks
    for chunk in tqdm(pd.read_csv("DATA/text_dataset.csv", chunksize=chunk_size)):
        preprocessed_review = preprocess_text(chunk['review'].values)
         saveDataFrame(pd.DataFrame({'preprocessed_review':preprocessed_review, 
               'target':chunk['target'].values
             }))
```


----------



## Python Performance

[How to Speed Up Pandas with Modin](https://towardsdatascience.com/how-to-speed-up-pandas-with-modin-84aa6a87bcdb)

[Speed Up Your Pandas Workflow with Modin](https://towardsdatascience.com/speed-up-your-pandas-workflow-with-modin-9a61acff0076)

[How we optimized Python API server code 100x](https://towardsdatascience.com/how-we-optimized-python-api-server-code-100x-9da94aa883c5)


## Scikit-learn Performance

[How to Speed up Scikit-Learn Model Training](https://medium.com/distributed-computing-with-ray/how-to-speed-up-scikit-learn-model-training-aaf17e2d1e1)


## Tensorflow Performance

[Optimizing a TensorFlow Input Pipeline: Best Practices in 2022](https://medium.com/@virtualmartire/optimizing-a-tensorflow-input-pipeline-best-practices-in-2022-4ade92ef8736)

[A simple guide to speed up your training in TensorFlow](https://blog.seeso.io/a-simple-guide-to-speed-up-your-training-in-tensorflow-2-8386e6411be4?gi=55c564475d16)

[Speed up your TensorFlow Training with Mixed Precision on GPUs and TPUs](https://towardsdatascience.com/speed-up-your-tensorflow-training-with-mixed-precision-on-gpu-tpu-acf4c8c0931c)

[Accelerate your training and inference running on Tensorflow](https://towardsdatascience.com/accelerate-your-training-and-inference-running-on-tensorflow-896aa963aa70)

[Leverage the Intel TensorFlow Optimizations for Windows to Boost AI Inference Performance](https://medium.com/intel-tech/leverage-the-intel-tensorflow-optimizations-for-windows-to-boost-ai-inference-performance-ba56ba60bcc4)



## Tensorflow GPU

[Using GPUs With Keras and wandb: A Tutorial With Code](https://wandb.ai/authors/ayusht/reports/Using-GPUs-With-Keras-A-Tutorial-With-Code--VmlldzoxNjEyNjE)

[Use a GPU with Tensorflow](https://www.tensorflow.org/guide/gpu)

[Using an AMD GPU in Keras with PlaidML](https://www.petelawson.com/post/using-an-amd-gpu-in-keras/)


[The Ultimate TensorFlow-GPU Installation Guide For 2022 And Beyond](https://towardsdatascience.com/the-ultimate-tensorflow-gpu-installation-guide-for-2022-and-beyond-27a88f5e6c6e)

[Time to Choose TensorFlow Data over ImageDataGenerator](https://towardsdatascience.com/time-to-choose-tensorflow-data-over-imagedatagenerator-215e594f2435)



## Tensorflow on macOs

[Getting Started with tensorflow-metal PluggableDevice](https://developer.apple.com/metal/tensorflow-plugin/)

[GPU-Accelerated Machine Learning on MacOS](https://towardsdatascience.com/gpu-accelerated-machine-learning-on-macos-48d53ef1b545)

[apple/tensorflow_macos](https://github.com/apple/tensorflow_macos/issues/153)

[Tensorflow Mac OS GPU Support](https://stackoverflow.com/questions/44744737/tensorflow-mac-os-gpu-support)

[Install Tensorflow 2 and PyTorch for AMD GPUs](https://medium.com/analytics-vidhya/install-tensorflow-2-for-amd-gpus-87e8d7aeb812)

[Installing TensorFlow on the M1 Mac](https://towardsdatascience.com/installing-tensorflow-on-the-m1-mac-410bb36b776)




## References

[1] [Why Python is so slow and how to speed it up](https://towardsdatascience.com/why-is-python-so-slow-and-how-to-speed-it-up-485b5a84154e

[2] [Lightning Fast Iteration Tips For Python Programmers](https://towardsdatascience.com/lightning-fast-iteration-tips-for-python-programmers-61d4f72bf4f0)

[3] [Optimizing Memory Usage in Python Applications](https://towardsdatascience.com/optimizing-memory-usage-in-python-applications-f591fc914df5)

[4] [Optimizing memory usage in Python code](https://medium.com/geekculture/optimising-memory-usage-in-python-code-d50a9c2a562b)

[5] [5 Tips To Optimize Your Python Code](https://towardsdatascience.com/try-these-5-tips-to-optimize-your-python-code-c7e0ccdf486a?source=rss----7f60cf5620c9---4)

[6] [Introduction to Intel oneAPI AI Analytics Toolkit](https://pub.towardsai.net/introduction-to-intels-oneapi-ai-analytics-toolkit-8dd873925b96?gi=25547ad4241c)


[7] [4 Python Libraries that make it easier to Work with Large Datasets](https://towardsdatascience.com/4-python-libraries-that-ease-working-with-large-dataset-8e91632b8791)

[8] [Pandas tips to deal with huge datasets](https://kvirajdatt.medium.com/pandas-tips-to-deal-with-huge-datasets-f6a012d4e953)

[9] [Top 2 tricks for compressing and loading huge datasets](https://medium.com/the-techlife/top-2-tricks-for-compressing-and-loading-huge-datasets-91a7e394c933)
