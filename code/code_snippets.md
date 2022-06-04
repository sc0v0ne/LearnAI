# Python Code Snippets

<!-- MarkdownTOC levels=1,2,3 -->

- Show samples from each class
- Display multiple images in one figure
- Plot images side by side
- Visualize a batch of image data
- Python one-liners
- Utility Classes
    - Enumeration
    - Data Classes
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



## References

[1] [Binary Image Classification in PyTorch](https://towardsdatascience.com/binary-image-classification-in-pytorch-5adf64f8c781)

[2] [Learn Python By Example: 10 Python One-Liners That Will Help You Save Time](https://medium.com/@alains/learn-python-by-example-10-python-one-liners-that-will-help-you-save-time-ccc4cabb9c68)

[3] [3 Alternatives for Regular Custom Classes in Python]()


[6 Must-Know Methods in Python’s Random Module](https://medium.com/geekculture/6-must-know-methods-in-pythons-random-module-338263b5f927)
