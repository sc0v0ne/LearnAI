# Pandas


<!-- MarkdownTOC -->

- Basics
- Convert to Best Data Types Automatically
- Dates
- Iteration
- String
- Indexes
- Functions
- Aggregate
- Pivot
- References

<!-- /MarkdownTOC -->

## Basics

[Practical Pandas Tricks - Part 1: Import and Create DataFrame](https://towardsdatascience.com/introduction-to-pandas-part-1-import-and-create-dataframe-e53326b6e2b1)

[4 Must-Know Parameters in Python Pandas](https://towardsdatascience.com/4-must-know-parameters-in-python-pandas-6a4e36f6ddaf)

[How To Change Column Type in Pandas DataFrames](https://towardsdatascience.com/how-to-change-column-type-in-pandas-dataframes-d2a5548888f8)


## Convert to Best Data Types Automatically

When we load data as Pandas dataframe, Pandas automatically assigns a datatype to the variables/columns in the dataframe, typically the datatypes would be int, float and object datatypes. With the recent Pandas 1.0.0, we can make Pandas infer the best datatypes for the variables in a dataframe.

We will use the Pandas `convert_dtypes()` function and convert the to best data types automatically. Another big advantage of using convert_dtypes() is that it supports Pandas new type for missing values `pd.NA`.

```py
    import pandas as pd

    # check version
    print(pd.__version__)

    data_url = "https://raw.githubusercontent.com/cmdlinetips/data/master/gapminder-FiveYearData.csv"
    df = pd.read_csv(data_url)

    df.dtypes

    df.convert_dtypes().dtypes
```

By default, convert_dtypes will attempt to convert a Series (or each Series in a DataFrame) to dtypes that support `pd.NA`. 

By using the options convert_string, convert_integer, and convert_boolean, it is possible to turn off individual conversions to StringDtype, the integer extension types or BooleanDtype, respectively.


## Dates

[11 Essential Tricks To Demystify Dates in Pandas](https://towardsdatascience.com/11-essential-tricks-to-demystify-dates-in-pandas-8644ec591cf1)

[Dealing With Dates in Pandas](https://towardsdatascience.com/dealing-with-dates-in-pandas-6-common-operations-you-should-know-1ea6057c6f4f)


## Iteration

[How To Loop Through Pandas Rows](https://cmdlinetips.com/2018/12/how-to-loop-through-pandas-rows-or-how-to-iterate-over-pandas-rows/amp/)


## String

[String Operations on Pandas DataFrame](https://blog.devgenius.io/string-operations-on-pandas-dataframe-88af220439d1)


## Indexes

[How To Convert a Column to Row Name/Index in Pandas](https://cmdlinetips.com/2018/09/how-to-convert-a-column-to-row-name-index-in-pandas/amp/)

[8 Quick Tips on Manipulating Index with Pandas](https://towardsdatascience.com/8-quick-tips-on-manipulating-index-with-pandas-c10ef9d1b44f)


## Functions

[apply() vs map() vs applymap() in Pandas](https://towardsdatascience.com/apply-vs-map-vs-applymap-pandas-529acdf6d744)

[How to Combine Data in Pandas](https://towardsdatascience.com/how-to-combine-data-in-pandas-5-functions-you-should-know-651ac71a94d6)


## Aggregate

[6 Lesser-Known Pandas Aggregate Functions](https://towardsdatascience.com/6-lesser-known-pandas-aggregate-functions-c9831b366f21)

[Pandas Groupby and Sum](https://cmdlinetips.com/2020/07/pandas-groupby-and-sum/amp/)


## Pivot

[5 Minute Guide to Pandas Pivot Tables](https://towardsdatascience.com/5-minute-guide-to-pandas-pivot-tables-df2d02786886)



## References

[How to Convert to Best Data Types Automatically in Pandas?](https://cmdlinetips.com/2020/04/how-to-convert-to-best-data-types-automatically-in-pandas/amp/)

