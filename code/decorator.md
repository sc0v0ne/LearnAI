# The Decorator Pattern

<!-- MarkdownTOC -->

- @staticmethod
- @classmethod
- @property
- Decorator Code Snippets
- Timer
- Measure Function Performance
- Repeat
- Show prompt
- Try/Catch
- Convert Data
- Memoization
- Function Catalog
- New Type Annotation Features in Python 3.11
- Self — the Class Type
- Arbitrary Literal String
- Varying Generics
- TypedDict — Flexible Key Requirements
- References

<!-- /MarkdownTOC -->

## @staticmethod

A static method is a method that does not require the creation of an instance of a class. 

```py
    class Cellphone:
        def __init__(self, brand, number):
            self.brand = brand
            self.number = number
            
        def get_number(self):
            return self.number
          
        @staticmethod
        def get_emergency_number():
            return "911"
          
    Cellphone.get_emergency_number()
    # '911'
```

## @classmethod

A class method requires the class itself as the first argument which is written as cls. 

A class method normally works as a factory method and returns an instance of the class with supplied arguments. However, it does not have to work as a factory class and return an instance.

We can create an instance in the class method and do whatever you need without having to return it.

Class methods are very commonly used in third-party libraries.

Here, it is a factory method here and returns an instance of the Cellphone class with the brand preset to “Apple”.

```py
    class Cellphone:
        def __init__(self, brand, number):
            self.brand = brand
            self.number = number
            
        def get_number(self):
            return self.number
          
        @staticmethod
        def get_emergency_number():
            return "911"
          
        @classmethod
        def iphone(cls, number):
            _iphone = cls("Apple", number)
            print("An iPhone is created.")
            return _iphone
         
    iphone = Cellphone.iphone("1112223333")
    # An iPhone is created.
    iphone.get_number()
    # "1112223333"
    iphone.get_emergency_number()
    # "911"
```

If you use class methods properly, you can reduce code redundancy dramatically and make your code more readable and more professional. 

The key idea is that we can create an instance of the class based on some specific arguments in a class method, so we do not have to repeatedly create instances in other places (DRY).


## @property

In the code snippet above, there is a function called `get_number` which returns the number of a Cellphone instance. 

We can optimize the method a bit and return a formatted phone number.


In Python, we can also use getter and setter to easily manage the attributes of the class instances.


```py
    class Cellphone:
        def __init__(self, brand, number):
            self.brand = brand
            self.number = number
            
        @property
        def number(self):
            _number = "-".join([self._number[:3], self._number[3:6],self._number[6:]])
            return _number
        
        @number.setter
        def number(self, number):
            if len(number) != 10:
                raise ValueError("Invalid phone number.")
            self._number = number

    cellphone = Cellphone("Samsung", "1112223333")
    print(cellphone.number)
    # 111-222-3333

    cellphone.number = "123"
    # ValueError: Invalid phone number.
```


Here is the complete example using the three decorators in Python: `@staticmethod`, `@classmethod`, and `@property`:

```py
    class Cellphone:
        def __init__(self, brand, number):
            self.brand = brand
            self.number = number
            
        @property
        def number(self):
            _number = "-".join([self._number[:3], self._number[3:6],self._number[6:]])
            return _number

        @number.setter
        def number(self, number):
            if len(number) != 10:
                raise ValueError("Invalid phone number.")
            self._number = number
        
        @staticmethod
        def get_emergency_number():
            return "911"
        
        @classmethod
        def iphone(cls, number):
            _iphone = cls("Apple", number)
            print("An iPhone is created.")
            return _iphone
```


## Decorator Code Snippets

## Timer

```py
    def timer(func):
      """
      Display time it took for our function to run. 
      """
      @wraps(func)
      def wrapper(*args, **kwargs):
        start = time.perf_counter()
    
        # Call the actual function
        res = func(*args, **kwargs)
    
        duration = time.perf_counter() - start
        print(f'[{wrapper.__name__}] took {duration * 1000} ms')
        return res
        return wrapper
```

```py
    @timer
    def isprime(number: int):
      """ Check if a number is a prime number """
      isprime = False
      for i in range(2, number):
        if ((number % i) == 0):
          isprime = True
          break
          return isprime
```

## Measure Function Performance

```py
    def performance_check(func):
        """ Measure performance of a function """
        @wraps(func)
        def wrapper(*args, **kwargs):
          tracemalloc.start()
          start_time = time.perf_counter()
          res = func(*args, **kwargs)
          duration = time.perf_counter() - start_time
          current, peak = tracemalloc.get_traced_memory()
          tracemalloc.stop()
    
          print(f"\nFunction:             {func.__name__} ({func.__doc__})"
                f"\nMemory usage:         {current / 10**6:.6f} MB"
                f"\nPeak memory usage:    {peak / 10**6:.6f} MB"
                f"\nDuration:             {duration:.6f} sec"
                f"\n{'-'*40}"
          )
          return res
          return wrapper
```

```py
    @performance_check
    def is_prime_number(number: int):
        """Check if a number is a prime number"""
        # ....rest of the function
```

## Repeat

```py
    def repeater(iterations:int=1):
      """ Repeat the decorated function [iterations] times """
      def outer_wrapper(func):
        def wrapper(*args, **kwargs):
          res = None
          for i in range(iterations):
            res = func(*args, **kwargs)
          return res
        return wrapper
        return outer_wrapper
```

```py
    @repeater(iterations=2)
    def sayhello():
      print("hello")
```

## Show prompt

```py
    def prompt_sure(prompt_text:str):
      """ Show prompt asking you whether you want to continue. Exits on anything but y(es) """
      def outer_wrapper(func):
        def wrapper(*args, **kwargs):
          if (input(prompt_text).lower() != 'y'):
            return
          return func(*args, **kwargs)
        return wrapper
        return outer_wrapper
```

```py
    @prompt_sure('Sure? Press y to continue, press n to stop')
    def say_hi():
      print("hi")
```

## Try/Catch

```py
    def trycatch(func):
      """ Wraps the decorated function in a try-catch. If function fails print out the exception. """
      @wraps(func)
      def wrapper(*args, **kwargs):
        try:
          res = func(*args, **kwargs)
          return res
        except Exception as e:
          print(f"Exception in {func.__name__}: {e}")
          return wrapper
```

```py
    @trycatch
    def trycatchExample(numA:float, numB:float):
      return numA / numB
```

## Convert Data

```py
import numpy as np
import pandas as pd
 
# function decorator to ensure numpy input
# and round off output to 4 decimal places
def ensure_numpy(fn):
    def decorated_function(data):
        array = np.asarray(data)
        output = fn(array)
        return np.around(output, 4)
    return decorated_function
 
@ensure_numpy
def numpysum(array):
    return array.sum()
 
x = np.random.randn(10,3)
y = pd.DataFrame(x, columns=["A", "B", "C"])
 
# output of numpy .sum() function
print("x.sum():", x.sum())
print()
 
# output of pandas .sum() funuction
print("y.sum():", y.sum())
print(y.sum())
print()
 
# calling decorated numpysum function
print("numpysum(x):", numpysum(x))
print("numpysum(y):", numpysum(y))
```

## Memoization

There are some function calls that we do repeatedly but the values rarely change. 

This could be calls to a server where the data is relatively static or as part of a dynamic programming algorithm or computationally intensive math function. 

We might want to memoize these function calls -- storing the value of their output on a virtual memo pad for reuse later.

A decorator is the best way to implement a memoization function. 

Here, we implement the `memoize()` to work using a global dictionary MEMO such that the name of a function together with the arguments becomes the key and the function’s return becomes the value. 

When the function is called, the decorator will check if the corresponding key exists in MEMO, and the stored value will be returned. Otherwise, the actual function is invoked and its return value is added to the dictionary.

```py
import pickle
import hashlib
 
 
MEMO = {} # To remember the function input and output
 
def memoize(fn):
    def _deco(*args, **kwargs):
        # pickle the function arguments and obtain hash as the store keys
        key = (fn.__name__, hashlib.md5(pickle.dumps((args, kwargs), 4)).hexdigest())
        # check if the key exists
        if key in MEMO:
            ret = pickle.loads(MEMO[key])
        else:
            ret = fn(*args, **kwargs)
            MEMO[key] = pickle.dumps(ret)
        return ret
    return _deco
 
@memoize
def fibonacci(n):
    if n in [0, 1]:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)
 
print(fibonacci(40))
print(MEMO)
```

Memoization is very helpful for expensive functions whose outputs do not change frequently such as reading stock market data from the Internet. 

```py
import pandas_datareader as pdr
 
@memoize
def get_stock_data(ticker):
    # pull data from stooq
    df = pdr.stooq.StooqDailyReader(symbols=ticker, start="1/1/00", end="31/12/21").read()
    return df
 
#testing call to function
import cProfile as profile
import pstats
 
for i in range(1, 3):
    print(f"Run {i}")
    run_profile = profile.Profile()
    run_profile.enable()
    get_stock_data("^DJI")
    run_profile.disable()
    pstats.Stats(run_profile).print_stats(0)
```

Python 3.2 or later shipped you the decorator lru_cache from the built-in library functools. 

The lru_cache implements LRU caching, which limits its size to the most recent calls (default 128) to the function. In Python 3.9, there is a @functools.cache as well, which is unlimited in size without the LRU purging.

```py
import functools
import pandas_datareader as pdr
 
# memoize using lru_cache
@functools.lru_cache
def get_stock_data(ticker):
    # pull data from stooq
    df = pdr.stooq.StooqDailyReader(symbols=ticker, start="1/1/00", end="31/12/21").read()
    return df
 
# testing call to function
import cProfile as profile
import pstats
 
for i in range(1, 3):
    print(f"Run {i}")
    run_profile = profile.Profile()
    run_profile.enable()
    get_stock_data("^DJI")
    run_profile.disable()
    pstats.Stats(run_profile).print_stats(0)
```


## Function Catalog

Another example is to register functions in a catalog which allows us to associate functions with a string and pass the strings as arguments for other functions. 

A function catalog is the start to making a system to allow user-provided plug-ins such `activate()`.


----------



## New Type Annotation Features in Python 3.11

The improvement of type annotations in Python 3.11 can help to write bug-free code.

## Self — the Class Type

The following code does not use type hints which may cause problems. 

```py
class Box:
    def paint_color(self, color):
        self.color = color
        return self
```

We can use Self to indicate that the return value is an object in the type of “Self" which is interpreted as the Box class.

```py
from typing import Self
class Box:
    def paint_color(self, color: str) -> Self:
        self.color = color
        return self
```

## Arbitrary Literal String

When we want a function to take a string literal, we must specify the compatible string literals. 
        
Python 3.11 introduces a new general type named `LiteralString` which allows the users to enter any string literals. 

```py
from typing import LiteralString


def paint_color(color: LiteralString):
    pass


paint_color("cyan")
paint_color("blue")
```

The `LiteralString` type gives the flexibility of using any string literals instead of specific string literals when we use the `Literal` type. 

## Varying Generics

We can use `TypeVar` to create generics with a single type, as we did previously for Box. When we do numerical computations (such as array-based operations in NumPy and TensorFlow), we use arrays that have varied dimensions and shapes.

When we provide type annotations to these varied shapes, it can be cumbersome to provide type information for each possible shape which requires a separate definition of a class since the exiting TypeVar can only handle a single type at a time.

Python 3.11 is introducing the TypeVarTuple that allows you to create generics using multiple types. Using this feature, we can refactor our code in the previous snippet, and have something like the below:

```py
from typing import Generic, TypeVarTuple
Dim = TypeVarTuple('Dim')
class Shape(Generic[*Dim]):
    pass
```

Since it is a tuple object, we can use a starred expression to unpack its contained objects which is a variable number of types. 

The above Shape class can be of any shape which has more flexibility and eliminates the need of creating separate classes for different shapes.

## TypedDict — Flexible Key Requirements

In Python, dictionaries are a powerful data type that saves data in the form of key-value pairs. 

The keys are arbitrary and you can use any applicable keys to store data. However, sometimes we may want to have a structured dictionary that has specific keys and the values of a specific type which means using TypedDict. 

```py
from typing import TypedDict
class Name(TypedDict):
    first_name: str
    last_name: str
```

We know that some people may have middle names (middle_name) and some do not. 

There are no direct annotations to make a key optional and the current workaround is creating a superclass that uses all the required keys while the subclass includes the optional keys. 

Python 3.11 introduces NotRequired as a type qualifier to indicate that a key can be potentially missing for TypedDict. The usage is very straightforward. 

```py
from typing import TypedDict, NotRequired
class Name(TypedDict):
    first_name: str
    middle_name: NotRequired[str]
    last_name: str
```

If we have too many optional keys, we can specify those keys that are required using `Required` instead of specifying those optional as not required. 

Thus, the alternative equivalent solution for the above issue:

```py
from typing import TypedDict, Required
class Name(TypedDict, total=False):
    first_name: Required[str]
    middle_name: str
    last_name: Required[str]
```

Note in the code snippet we specify `total=False` which makes all the keys optional. In the meantime, we mark these required keys as `Required` which means that the other keys are optional.



## References

[A Gentle Introduction to Decorators in Python](https://machinelearningmastery.com/a-gentle-introduction-to-decorators-in-python/)

[5 real handy python decorators for analyzing/debugging your code](https://towardsdatascience.com/5-real-handy-python-decorators-for-analyzing-debugging-your-code-c22067318d47)
 
[How to Use the Magical @staticmethod, @classmethod, and @property Decorators in Python](https://betterprogramming.pub/how-to-use-the-magical-staticmethod-classmethod-and-property-decorators-in-python-e42dd74e51e7?gi=8734ec8451fb)

[4 New Type Annotation Features in Python 3.11](https://betterprogramming.pub/4-new-type-annotation-features-in-python-3-11-84e7ec277c29)

