# Memory Usage Tips

<!-- MarkdownTOC levels=1,2,3 -->

- Effective use of Data Types
    - Numerical Features
    - DateTime
    - Categorical
    - Creating dummy variables for modeling
    - Converting between String and Datetime
    - Typecasting while Reading Data
- Optimizing Memory Usage in Python Applications
    - Find Bottlenecks
    - Saving Some RAM
    - Not Using RAM At All
- How to Profile Memory Usage
- References

<!-- /MarkdownTOC -->

Here are some resources to evaluate Python memory usage, also see [Performance](./performance.md). 



## Effective use of Data Types

Make effective use of data types to prevent crashing of memory.

When the size of the dataset is comparatively larger than memory using such libraries is preferred, but when dataset size comparatively equal or smaller to memory size, we can optimize the memory usage while reading the dataset. 

Here, we discuss how to optimize memory usage while loading the dataset using `read_csv()`.

Using `df.info()` we can view the default data types and memory usage.


The default list of data types assigned by Pandas are:

|  dtype        |                 Usage                  |
| :------------ | :------------------------------------- |
| object        | Text or mixed numeric and text values  |
| int64         | Integer numbers                        |
| float64       | Floating-point numbers                 |
| bool          | True/False values                      |
| datetime64    | Date and time values                   |
| timedelta[ns] | Difference between two datetime values |
| category      | Finite list of text values             |


### Numerical Features

For all numerical values, Pandas assigns float64 data type to a feature column having at least one float value, and int64 data type to a feature column having all feature values as integers.

Here is a list of the ranges of each datatype:

|  Data Type    |                            Description                                   |
| :------------ | :----------------------------------------------------------------------- |
| bool_         | Boolean stored as a byte                                                 |
| int_          | Default integer type (same as C long)                                    |
| intc          | Same as C int (int32 or int64)                                           |
| intp          | Integer used for indexing (ssize_t - int32 or int64)                     |
| int8          | Integer (-2^7 to 2^7 - 1)                                                |
| int16         | Integer (-2^15 to 2^15 - 1)                                              |
| intn          | Integer (-2^(n-1) to 2^(n-1) - 1)                                        |
| uint8         | Unsigned integer (0 to 255)                                              |
| uint16        | Unsigned integer (0 to 2^16 - 1)                                         |
| uint32        | Unsigned integer (0 to 2^32 - 1)                                         |
| float_        | Shorthand for float64                                                    |
| float16       | Half-precision float: sign bit, 5-bit exponent, and 10-bit mantissa    |
| float32       | Single-precision float: sign bit, 8-bit exponent, and 32-bit mantissa  |
| float64       | Double-precision float: sign bit, 11-bit exponent, and 52-bit mantissa |
| complex_      | Shorthand for complex128                                                 |
| complex64     | Complex number represented by two 32-bit floats                          |
| complex128     | Complex number represented by two 64-bit floats                          |


NOTE: A value with data type as int8 takes 8x times less memory compared to int64 data type.

### DateTime

By default, datetime columns are assigned as object data type that can be downgraded to DateTime format.

### Categorical

Pandas assign non-numerical feature columns as object data types which can be downgraded to category data types.

The non-numerical feature column usually has categorical variables which are mostly repeating. 

For example, the gender feature column has just 2 categories ‘Male’ and ‘Female’ that are repeating over and over again for all the instances which are re-occupying the space. 

Assigning gender to category datatype is a more compact representation.

**Better performance with categoricals**

If you do a lot of analytics on a particular dataset, converting to categorical can yield substantial overall performance gains. 

A categorical version of a DataFrame column will often use significantly less memory.

#### Better performance with categoricals

If you do a lot of analytics on a particular dataset, converting to categorical can yield substantial overall performance gains. A categorical version of a DataFrame column will often use significantly less memory, too.

#### Categorical Methods

Series containing categorical data have several special methods similar to the `Series.str` specialized string methods. This also provides convenient access to the categories and codes. 

The special attribute cat provides access to categorical methods:

```py
  s = pd.Series(['a', 'b', 'c', 'd'] * 2)
  
  cat_s = s.astype('category')
  cat_s.cat.codes
  cat_s.cat.categories
  
  
  actual_categories = ['a', 'b', 'c', 'd', 'e']

  cat_s2 = cat_s.cat.set_categories(actual_categories)
  cat_s2.value_counts()
```

In large datasets, categoricals are often used as a convenient tool for memory savings and better performance. After you filter a large DataFrame or Series, many of the categories may not appear in the data.

```py
  cat_s3 = cat_s[cat_s.isin(['a', 'b'])]
  cat_s3.cat.remove_unused_categories()
```

Table 12-1: Categorical methods for Series in pandas

### Creating dummy variables for modeling

When using statistics or machine learning tools, we usually transform categorical data into dummy variables callwd _one-hot encoding_ which involves creating a DataFrame with a column for each distinct category; these columns contain 1s for occurrences of a given category and 0 otherwise.

```py
  cat_s = pd.Series(['a', 'b', 'c', 'd'] * 2, dtype='category')

  pd.get_dummies(cat_s)
```


### Converting between String and Datetime

You can format datetime objects and pandas Timestamp objects as strings using `str` or the `strftime` method passing a format specification. 

```py
    stamp = datetime(2011, 1, 3)

    str(stamp)
    stamp.strftime('%Y-%m-%d')
```

You can use many of the same format codes to convert strings to dates using `datetime.strptime`. 

```py
    value = '2011-01-03'

    datetime.strptime(value, '%Y-%m-%d')

    datestrs = ['7/6/2011', '8/6/2011']

    [datetime.strptime(x, '%m/%d/%Y') for x in datestrs]
```

pandas is generally oriented toward working with arrays of dates whether used as an axis index or a column in a DataFrame. 

The `to_datetime` method parses many different kinds of date representations. It also handles values that should be considered missing (None, empty string, etc.). 

NaT (Not a Time) is pandas’s null value for timestamp data.


12.2 Advanced GroupBy Use

12.3 Techniques for Method Chaining

The pipe Method

You can accomplish a lot with built-in pandas functions and the approaches to method chaining with callables that we just looked at. 

Sometimes you need to use your own functions or functions from third-party libraries. 

```py
    a = f(df, arg1=v1)
    b = g(a, v2, arg3=v3)
    c = h(b, arg4=v4)

    result = (df.pipe(f, arg1=v1)
            .pipe(g, v2, arg3=v3)
            .pipe(h, arg4=v4))
```

The statement `f(df)` and `df.pipe(f)` are equivalent but `pipe` makes chained invocation easier.


### Typecasting while Reading Data

The `read_csv` function includes a type parameter which accepts user-provided data types in a key-value format that can use instead of the default ones. 

The DateTime feature column can be passed to the `parse_dates` parameter.

```py
    dtype_dict = {
        'vendor_id': 'int8',
        'passenger_count': 'int8',
        'pickup_longitude': 'float16',
        'pickup_latitude': 'float16',
        'dropoff_longitude': 'float16',
        'dropoff_latitude': 'float16',
        'store-and_fwd_flag': 'category',
        'trip_duration': 'int32'
    }

    dates = ['pickup_datetime', 'dropoff_datetime']

    df = pd.read_csv("../data/train.csv",
                     dtype=dtype_dict,
                     parse_dates=dates)

    print(df.shape)
    print(df.info(verbose=False, memory_usage='deep'))
```


----------



## Optimizing Memory Usage in Python Applications

Fins out why your Python apps are using too much memory and reduce their RAM usage with these simple tricks and efficient data structures

### Find Bottlenecks

First we need to find the bottlenecks in the code that are hogging memory.

The `memory_profiler` tool measures memory usage of specific function on line-by-line basis. 

We also install the `psutil` package which significantly improves the profiler performance.

The memory_profiler shows memory usage/allocation on line-by-line basis for the decorated function (here the memory_intensive function) which intentionally creates and deletes large lists.

Now that we are able to find specific lines that increase memory consumption, we can see how much each variable is using. 

If we were to use `sys.getsizeof` to measure to measure variables, we woll get questionable information for some types of data structures. For integers or bytearrays we will get the real size in bytes, for containers such as list though, we will only get size of the container itself and not its contents. 

A better approach is to the pympler tool that is designed for analyzing memory behaviour which can help you get more realistic view of Python object sizes. 

```py
    from pympler import asizeof

    print(asizeof.asizeof([1, 2, 3, 4, 5]))
    # 256

    print(asizeof.asized([1, 2, 3, 4, 5], detail=1).format())
    # [1, 2, 3, 4, 5] size=256 flat=96
    #     1 size=32 flat=32
    #     2 size=32 flat=32
    #     3 size=32 flat=32
    #     4 size=32 flat=32
    #     5 size=32 flat=32

    print(asizeof.asized([1, 2, [3, 4], "string"], detail=1).format())
    # [1, 2, [3, 4], 'string'] size=344 flat=88
    #     [3, 4] size=136 flat=72
    #     'string' size=56 flat=56
    #     1 size=32 flat=32
    #     2 size=32 flat=32
```

Pympler provides `asizeof` module with function of same name which correctly reports size of the list as well all values it contains and the `asized` function which can give a more detailed size breakdown of individual components of the object.

Pympler has many more features including tracking class instances or identifying memory leaks.  

### Saving Some RAM

Now we need to find a way to fix memory issues. The quickest and easiest solution can be switching to more memory-efficient data structures.

Python lists are one of the more memory-hungry options when it comes to storing arrays of values:

In this example we used the `array` module which can store primitives such as integers or characters. 

We can see that in this case memory usage peaked at just over 100MiB which is a huge difference in comparison to list. 

We can further reduce memory usage by choosing appropriate precision:

One major downside of using array as data container is that it doesn't support that many types.

If we need to perform a lot of mathematical operations on the data then should use NumPy arrays instead:

The above optimizations help with overall size of arrays of values but we can also improvem the size of individual objects defined by Python classes using `__slots__` class attribute which is used to explicitly declare class properties. 

Declaring `__slots__` on a class also prevents creation of `__dict__` and `__weakref__` attributes which can be useful:

How do we store strings depends on how we want to use them. If we are going to search through a huge number of string values then using `list` is a bad idea. 

The best option may be to use optimized data structure such as trie, especially for static data sets which you use for example for querying and there is a library for that as well as for many other tree-like data structures called [pytries](https://github.com/pytries).

### Not Using RAM At All

Perhaps the easiest way to save RAM is to not use memory in a first place. We obviously cannot avoid using RAM completely but we can avoid loading the full data set at once and work with the data incrementally when possible. 

The simplest way to achieve this is using generators which return a lazy iterator which computes elements on demand rather than all at once.

An even stronger tool that we can leverage is _memory-mapped files_ which allows us to load only parts of data from a file. 

The Python standard library provides `mmap` module which can be used to create memory-mapped files that behave  like both files and bytearrays that can be used with file operations such read, seek, or write as well as string operations:

```py
    import mmap

    with open("some-data.txt", "r") as file:
        with mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as m:
            print(f"Read using 'read' method: {m.read(15)}")
            # Read using 'read' method: b'Lorem ipsum dol'
            m.seek(0)  # Rewind to start
            print(f"Read using slice method: {m[:15]}")
            # Read using slice method
```

Loading and reading memory-mapped files is rather simple:

Most of the time, we will probably want to read the file as shown above but we also write to the memory-mapped file:

If we are performing computations in NumPy, it may he better to use its `memmap` feature which is suitable for NumPy arrays stored in binary files.


----------



## How to Profile Memory Usage

[How Much Memory is your ML Code Consuming?](https://towardsdatascience.com/how-much-memory-is-your-ml-code-consuming-98df64074c8f)

Learn how to quickly check the memory footprint of your machine learning function/module with one line of command. Generate a nice report too.

Monitor line-by-line memory usage of functions with memory profiler module. 

```bash
  pip install -U memory_profiler
  
  python -m memory_profiler some-code.py
```

It is easy to use this module to track the memory consumption of the function. `@profile` decorator that can be used before each function that needs to be tracked which will track the memory consumption line-by-line in the same way as of line-profiler.

```py
    from memory_profiler import profile
  
    @profile
    def my_func():
        a = [1] * (10 ** 6)
        b = [2] * (2 * 10 ** 7)
        c = [3] * (2 * 10 ** 8)
        del b
        return a    
      
    if __name__=='__main__':
        my_func()
```



## References

[1] [Optimize Pandas Memory Usage for Large Datasets](https://towardsdatascience.com/optimize-pandas-memory-usage-while-reading-large-datasets-1b047c762c9b)

[2] [Profile Memory Consumption of Python functions in a single line of code](https://towardsdatascience.com/profile-memory-consumption-of-python-functions-in-a-single-line-of-code-6403101db419)


