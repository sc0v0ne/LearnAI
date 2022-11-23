# Pandas Code Snippets


## Code Snippets

Here are some useful python code snippets for data science and analysis projects [1].

```py
    import pandas pd
    
    # Combine multiple excel sheets of the same file 
    # into single dataframe and save as .csv
    excel_file = pd.read_excel(‘file.xlsx’, sheet_name=None)
    dataset_combined = pd.concat(excel_file.values())

    # Extract the year from a date data In pandas
    df['year'] = df['date'].dt.year if we put dt.month is for month etc..
    df.head()
```


### Find boundaries for outliers

```py
    def find_boundaries(df, variable, distance=1.5):
    
        IQR = df[variable].quantile(0.75) - df[variable].quantile(0.25)
    
        lower_boundary = df[variable].quantile(0.25) - (IQR * distance)
        upper_boundary = df[variable].quantile(0.75) + (IQR * distance)
    
        return upper_boundary, lower_boundary
```


### Compute cardinality of datasets

```py
    data.nunique().plot.bar(figsize=(12,6))
    plt.ylabel('Number of unique categories')
    plt.xlabel('Variables')
    plt.title('Cardinality')
    
    ## Version with 5% threshold
    
    fig = label_freq.sort_values(ascending=False).plot.bar()
    fig.axhline(y=0.05, color='red')
    fig.set_ylabel('percentage of cars within each category')
    fig.set_xlabel('Variable: class')
    fig.set_title('Identifying Rare Categories')
    plt.show()
```

### Find Missing data with charts

```py
    data.isnull().mean().plot.bar(figsize=(12,6))
    plt.ylabel('Percentage of missing values')
    plt.xlabel('Variables')
    plt.title('Quantifying missing data')
```

### Add categorical/text labels

```py
    # Add column that gives a unique number to each of these labels 
    df['label_num'] = df['label'].map({
        'Household' : 0, 
        'Books': 1, 
        'Electronics': 2, 
        'Clothing & Accessories': 3
    })
    
    # check the results 
    df.head(5)
```

### Preprocess text with Spacy

```py
    import spacy
    
    # load english language model and create nlp object from it
    nlp = spacy.load("en_core_web_sm") 
    
    def preprocess(text):
        """
        utlity function for pre-processing text
        """
        # remove stop words and lemmatize the text
        doc = nlp(text)
        filtered_tokens = []
        for token in doc:
            if token.is_stop or token.is_punct:
                continue
            filtered_tokens.append(token.lemma_)
        
        return " ".join(filtered_tokens) 
    
        df['preprocessed_txt'] = df['Text'].apply(preprocess)
```



## Pandas one-liners

Here are some helpful Pandas one-liners [2]:

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




## References

[1] [8 useful python code snippets for data science and analysis projects](https://medium.com/mlearning-ai/8-useful-python-code-snippets-for-data-science-and-analysis-projects-e76b0f391fb)

[2] [Powerful One-liners in Pandas Every Data Scientist Should Know](https://towardsdatascience.com/powerful-one-liners-in-pandas-every-data-scientist-should-know-737e721b81b6)

