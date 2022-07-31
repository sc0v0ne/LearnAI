# Python Code Snippets

<!-- MarkdownTOC levels=1,2,3 -->

- Show samples from each class
- Display multiple images in one figure
- Plot images side by side
- Visualize a batch of image data
- Python one-liners
- Pandas one-liners
- Utility Classes
    - Enumeration
    - Data Classes
- Iterables
- Write Shorter Conditionals using Dictionaries
- Display Pandas DataFrame in table style
- References

<!-- /MarkdownTOC -->


## Show samples from each class

```py
    import numpy as np
    import matplotlib.pyplot as plt
    
    def show_images(num_classes):
        """
        Show image samples from each class
        """
        fig = plt.figure(figsize=(8,3))
    
        for i in range(num_classes):
            ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])
            idx = np.where(y_train[:]==i)[0]
            x_idx = X_train[idx,::]
            img_num = np.random.randint(x_idx.shape[0])
            im = np.transpose(x_idx[img_num,::], (1, 2, 0))
            ax.set_title(class_names[i])
            plt.imshow(im)
    
        plt.show()
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    num_train, img_channels, img_rows, img_cols =  X_train.shape
    num_test, _, _, _ =  X_train.shape
    num_classes = len(np.unique(y_train))
    
    class_names = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']
    
    show_images(num_classes)
```


## Display multiple images in one figure

```py
    # import libraries
    import cv2
    from matplotlib import pyplot as plt
      
    # create figure
    fig = plt.figure(figsize=(10, 7))
      
    # setting values to rows and column variables
    num_rows = 2
    num_cols = 2
    
    # Read the images into list
    images = []
    img = cv2.imread('Image1.jpg')
    images.append(img)
    
    img = cv2.imread('Image2.jpg')
    images.append(img)
    
    img = cv2.imread('Image3.jpg')
    images.append(img)
    
    img = cv2.imread('Image4.jpg')
    images.append(img)
    
      
    # Adds a subplot at the 1st position
    fig.add_subplot(num_rows, num_cols, 1)
      
    # showing image
    plt.imshow(Image1)
    plt.axis('off')
    plt.title("First")
      
    # Adds a subplot at the 2nd position
    fig.add_subplot(num_rows, num_cols, 2)
      
    # showing image
    plt.imshow(Image2)
    plt.axis('off')
    plt.title("Second")
      
    # Adds a subplot at the 3rd position
    fig.add_subplot(num_rows, num_cols, 3)
      
    # showing image
    plt.imshow(Image3)
    plt.axis('off')
    plt.title("Third")
      
    # Adds a subplot at the 4th position
    fig.add_subplot(num_rows, num_cols, 4)
      
    # showing image
    plt.imshow(Image4)
    plt.axis('off')
    plt.title("Fourth")
```


## Plot images side by side

```py
    _, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
    plt.show()
```


## Visualize a batch of image data

TODO: Add code sample


----------



## Python one-liners

Here are some helpful python one-liners that can save time [3]:

```py
    # Palindrome Python One-Liner
    phrase.find(phrase[::-1])

    # Swap Two Variables Python One-Liner
    a, b = b, a

    # Sum Over Every Other Value Python One-Liner
    sum(stock_prices[::2])

    # Read File Python One-Liner
    [line.strip() for line in open(filename)]

    # Factorial Python One-Liner
    reduce(lambda x, y: x * y, range(1, n+1))

    # Performance Profiling Python One-Liner
    python -m cProfile foo.py

    # Superset Python One-Liner
    lambda l: reduce(lambda z, x: z + [y + [x] for y in z], l, [[]])

    # Fibonacci Python One-Liner
    lambda x: x if x<=1 else fib(x-1) + fib(x-2)

    # Quicksort Python One-liner
    lambda L: [] if L==[] else qsort([x for x in L[1:] if x< L[0]]) + L[0:1] + qsort([x for x in L[1:] if x>=L[0]])

    # Sieve of Eratosthenes Python One-liner
    reduce( (lambda r,x: r-set(range(x**2,n,x)) if (x in r) else r), range(2,int(n**0.5)), set(range(2,n)))
```


## Pandas one-liners

Here are some helpful Pandas one-liners [6]:

```py
  # n-largest values in a series
  # find the top-n paid roles 
  data.nlargest(n, "Employee Salary", keep = "all")
  
  # n-smallest values in a series
  data.nsmallest(n, "Employee Salary", keep = "all")
```

```py
  # Crosstab computes a cross-tabulation of two (or more) columns/series and returns a frequency of each combination
  # compute the number of employees working from each location within every company
  pd.crosstab(data["Company Name"], data["Employee Work Location"])
  result_crosstab = pd.crosstab(data["Company Name"], data["Employee Work Location"])
  sns.heatmap(result_crosstab, annot=True)
  
  # compute aggregation on average salary
  result_crosstab = pd.crosstab(index = data["Company Name"], 
                columns=data["Employment Status"], 
                values = data["Employee Salary"], 
                aggfunc=np.mean)
  sns.heatmap(result_crosstab, annot=True, fmt='g')
  
  
  # Similar to crosstabs, pivot tables in Pandas provide a way to cross-tabulate your data.
  pd.pivot_table(data, 
               index=["Company Name"], 
               columns=["Employee Work Location"], 
               aggfunc='size', 
               fill_value=0)

  result_pivot = pd.pivot_table(data, 
            index=["Company Name"], 
            columns=["Employee Work Location"], 
            aggfunc='size', 
            fill_value=0)
               
  sns.heatmap(result_pivot, annot=True, fmt='g')
```

```py
  # Mark duplicate rows
  # Marks all duplicates as True except for the first occurrence.
  new_data.duplicated(keep="first")
  
  # create filtered Dataframe with no duplicates
  # Marks all duplicates as True
  new_data[~new_data.duplicated(keep=False)]
  
  # check duplicates on a subset of columns
  new_data.duplicated(subset=["Company Name", "Employee Work Location"], keep=False)
  
  # Remove duplicates
  new_data.drop_duplicates(keep="first")
  
  # drop duplicates on a subset of columns
  new_data.drop_duplicates(subset=["Company Name", "Employee Work Location"], keep = False)
  view raw
```


---------



## Utility Classes

### Enumeration

We can create an Enumeration class to hold related members of the same concept such as compass directions (north, south, east, and west) or seasons [3]. 

In the standard library of Python, the `enum` module provides the essential functionalities for creating an enumeration class.


```py
    from enum import Enum
    
    class Season(Enum):
        SPRING = 1
        SUMMER = 2
        FALL = 3
        WINTER = 4
```

```py
    spring = Season.SPRING
    spring.name
    spring.value
    
    fetched_season_value = 2
    matched_season = Season(fetched_season_value)
    matched_season
    # <Season.SUMMER: 2>
    
    list(Season)
    
    [x.name for x in Season]
```

### Data Classes

We can a class to hold data using the dataclass decorator [3]. 

```py
    from dataclasses import dataclass
    
    @dataclass
    class Student:
        name: str
        gender: str
```

```py
    student = Student("John", "M")
    student.name
    student.gender
    
    repr(student)   # __repr__
    # "Student(name='John', gender='M')"
    
    print(student)  # __str__
    Student(name='John', gender='M')
```


## Iterables

An _iterable_ is any Python object that is capable of returning its members one at a time, permitting it to be iterated over in a loop.

There are _sequential_ iterables that arrange items in a specific order, such as lists, tuples, string and dictionaries.

There are _non-sequential_ collections that are iterable. For example, a set is an iterable, despite lacking any specific order.

Iterables are fundamental to Python and manuy other programming language, so knowing how to efficiently use them will have an impact on the quality of your code.

In general, iterables can be processed using a for-loop that allows for successively handling each item that is part of the iterable.

You may find that you often create programming loops that are similar to one another. Therefore, there are plenty of great built-in Python functions that can help to process iterables without reinventing the wheel.

Python iterables are fast, memory-efficient, and when used properly make your code more concise and readable.

> Python’s built-in functions are written in C, so they are very fast and efficient. 


The functions included in the article [4] are:

- Function 1: all
- Function 2: any
- Funciton 3: enumerate
- Function 4: filter
- Function 5: map
- Function 6: min
- Function 7: max
- Function 8: reversed
- Function 9: sum
- Function 10: zip
- collections.Counter



## Write Shorter Conditionals using Dictionaries

Dictionaries are a concise alternative to the classic If-Else statement and the new Match-Case statement [5]. 

If we were to use an If-Else:

```py
    def month(idx):
        if idx == 1:
            return "January"
        elif idx == 2:
            return "February"
        # ...
        else:
        return "not a month"
```

Another approach would be to use the recently released Match-Case statement:

```py
    def month(idx):
        match idx:
            case 1:
                return "January"
            case 2:
                return "February"
            # ...
            case _:
        return "Not a month"
```

There is nothing inherently wrong with either of these approaches. In terms of code clarity, there is more to be gained using dictionaries.

Dictionary Approach

The first step is to return a dictionary that uses the index of the months as keys, and their corresponding names as values.

Next, we use the `.get()` method to obtain the name of the month that actually belongs to the number that we provided as function argument.

The great thing about this method is that we can also specify a default return value for when the requested key is not part of the dictionary (“not a month”).

```py
    def month(idx):
        return {
            1: "January",
            2: "February",
            # ...
            }.get(idx, "not a month")
```

While our dictionary conditional has the least-best performance, it is important to understand where this difference comes from.

Since we define our dictionary _within_ the `month()` function, it has to be constructed once for every function call which is inefficient. 

If we define the dictionary _outside_ the function and rerun the experiment, we achieve the fastest runtime.

```py
    dt = {
        1: "January",
        2: "February",
        # ...
        11: "November",
        12: "December"
    }

    def month(idx):
        return dt.get(idx, "not a month")
```



## Display Pandas DataFrame in table style

```py
    # importing the modules
    from tabulate import tabulate
    import pandas as pd
      
    # creating a DataFrame
    dict = {'Name':['Martha', 'Tim', 'Rob', 'Georgia'],
            'Maths':[87, 91, 97, 95],
            'Science':[83, 99, 84, 76]}
    df = pd.DataFrame(dict)
      
    # displaying the DataFrame
    print(tabulate(df, headers = 'keys', tablefmt = 'psql'))
```



## References

[1] [Binary Image Classification in PyTorch](https://towardsdatascience.com/binary-image-classification-in-pytorch-5adf64f8c781)

[2] [Learn Python By Example: 10 Python One-Liners That Will Help You Save Time](https://medium.com/@alains/learn-python-by-example-10-python-one-liners-that-will-help-you-save-time-ccc4cabb9c68)

[3] [3 Alternatives for Regular Custom Classes in Python]()

[4] [Iterator Functions](https://itnext.io/iterator-functions-33265a99e5d1)

[5] [Write Shorter Conditionals (Using Dictionaries)](https://itnext.io/write-shorter-conditionals-using-dictionaries-python-snippets-4-f92c8ce5eb7)

[6] [Powerful One-liners in Pandas Every Data Scientist Should Know](https://towardsdatascience.com/powerful-one-liners-in-pandas-every-data-scientist-should-know-737e721b81b6)


[Python: Pretty Print a Dict (Dictionary) – 4 Ways](https://datagy.io/python-pretty-print-dictionary/)

[Name of a Python function](https://medium.com/@vadimpushtaev/name-of-python-function-e6d650806c4)

[Python's assert: Debug and Test Your Code Like a Pro](https://realpython.com/python-assert-statement/)

[6 Must-Know Methods in Python’s Random Module](https://medium.com/geekculture/6-must-know-methods-in-pythons-random-module-338263b5f927)
