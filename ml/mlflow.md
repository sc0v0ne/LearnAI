# MLflow

<!-- MarkdownTOC -->

- What is MLflow?
- What is MLflow Tracking?
- Tracking Experiments with MLflow Tracking
    - Breaking Runs Down into Experiments
    - Logging experiments manually
    - Logging experiments automatically
- Packaging Projects with MLflow Projects
    - What are MLflow Projects?
    - Packaging Python Code for Conda Environments with Projects
    - Run Packaged MLflow Projects
- What are MLflow Models?
    - Saving Models with MLflow Models
    - Providing an input signature and input examples
    - Inferring the input signature automatically
    - Specifying the Input Signature Manually
    - Provide input examples
    - Save the model
    - Log the Model as an Artifact
    - Load the saved models
    - Inference with the Loaded models
    - Serve Models with MLflow Models
    - Inference with a Served Model
- Build a Python API and integrate it into a React web applicatio
    - Overview of the Application
    - Building the API
    - Building the API endpoints
- ML Pipelines
    - Ploomber
    - PyCaret
- References

<!-- /MarkdownTOC -->

## What is MLflow?

**MLflow** is an open-source platform for managing machine learning lifecycles that is designed to help data scientists and ML engineers facilitate the tracking of experiments and the deployment of code to a wide range of environments.

MLflow consists of the following four  components:

- MLflow Tracking: allows recording of experiments including the tracking of  models, hyperparameters, and artifacts.

- MLflow Projects: package data science code in a reproducible format.

- MLflow Models: export and deploy machine learning models in various environments.

- MLflow Registry: enables model storage, versioning, staging, and annotation.


## What is MLflow Tracking?

MLflow Tracking is a toolset for running and tracking machine learning experiments. 

Tracking relies on the concept of runs to organize and store tracking data. 

Each run records the following information:

MLflow also allows runs to be grouped into experiments. If you are going to be performing many tracking runs, you should break them up into experiments to keep everything neat and organized.

MLflow Tracking allows runs and artifacts to be recorded on:

- A local machine. This is the option we’ll be exploring in the guide below.
- A local machine with SQLite.
- A local machine with Tracking Server to listen to REST calls.
- Remote machines with Tracking Server.

We can view tracked data using the MLflow user interface.

Tracking can be done either manually or automatically. 

- With manual tracking, we log parameters, metrics, and artifacts by calling the associated functions and passing the values of interest to them. 

- MLflow also has built-in automatic loggers that record a number of predefined pieces of data for each of the supported packages.

## Tracking Experiments with MLflow Tracking

By default, Tracking records runs in a local mlruns directory where you run the project. We’re going to be using the default directory to keep things simple. You can check the current tracking directory by calling `mlflow.tracking.get_tracking_uri()`.

If we want to change the tracking directory, we use `mlflow.set_tracking_uri()` and pass a local or remote path where we want to use to store the data. We also need to prefix the path with `file:/`. 

```py
    import mlflow
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, train_test_split
    from sklearn import datasets
```

### Breaking Runs Down into Experiments

To keep many runs organized and easily scannable, we can group them into _experiments_.

By default, all runs are grouped into an experiment named “Default” which can be chaned using `mlflow.create_experiment()`.

- If we set an experiment that does not exist, MLflow will create the experiment for us.

- Keep in mind that deleted experiments are stored in mlruns/.trash, so you can recover them if necessary.

- Runs performed under an experiment are stored in the directory of that experiment under mlruns. Although we assign names to experiments when creating them, the experiment directory names reflect their IDs.

- When we launch MLflow, it creates a default experiment with the ID of 0; runs for this experiment are stored in mlruns/0; as we create experiments, the IDs are incremented.

```py
    # Creating an experiment
    experiment_id = mlflow.create_experiment(name="test")

    # Select an existing experiment
    mlflow.set_experiment(experiment_name="test")

    # Delete an experiment
    mlflow.delete_experiment(experiment_id=experiment_id)
```

### Logging experiments manually

MLflow allows you to track experiments either manually or automatically. 

We start with manual tracking using scikit-learn’s `LogisticRegression`.

```py
    # Set an experiment for manual logging
    mlflow.set_experiment(experiment_name="manual_logging")
```

To track data with MLflow Tracking, we can use the following functions:

- mlflow.start_run(): starts a new run and returns an mlflow.ActiveRun object, which can be used as a context manager within a Python with block. If a run is currently active, mlflow.start_run() returns it instead. You don’t need to start runs explicitly — calling any of the logging functions (listed below) starts a run automatically when no run is active.

- mlflow.end_run(): ends the currently active run. If you aren’t using a with block to leverage mlflow.ActiveRun as a context manager, you must call mlflow.end_run() after you are done with logging to terminate the run.

- mlflow.log_param(): logs a single key-value param, both stored as strings. For batch logging, use mlflow.log_params().

- mlflow.log_metric(): logs a single key-value metric, where the value must be a number. For batch logging, use mlflow.log_metrics().

- mlflow.set_tag(): sets a single key-value tag. For batch tagging, use mlflow.set_tags().

- mlflow.log_artifact(): log a local file or directory as an artifact. Batch logging is done with mlflow.log_artifacts().


Here is a full example:

```py
    # Checking if the script is executed directly
    if __name__ == "__main__":
        # Loading data
        data = datasets.load_breast_cancer()
        
        # Splitting the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(data.data, 
                                                            data.target,
                                                            stratify=data.target)
        
        # Selecting a parameter range to try out
        C = list(range(1, 10))
        
        # Starting a tracking run
        with mlflow.start_run(run_name="PARENT_RUN"):
            # For each value of C, running a child run
            for param_value in C:
                with mlflow.start_run(run_name="CHILD_RUN", nested=True):
                    # Instantiating and fitting the model
                    model = LogisticRegression(C=param_value, max_iter=1000)            
                    model.fit(X=X_train, y=y_train)
                    
                    # Logging the current value of C
                    mlflow.log_param(key="C", value=param_value)
                    
                    # Logging the test performance of the current model                
                    mlflow.log_metric(key="Score", value=model.score(X_test, y_test)) 
                    
                    # Saving the model as an artifact
                mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
```

If we run the code, experiment results will be logged to the `mlruns` path in your project directory.

We do not have to check out the results manually in the logging dir -- we can use MLflow’s UI to more conveniently view your experiments.

To launch MLflow’s user interface, navigate to the directory above `mlruns` in the terminal and run the following:

```bash
    mlflow ui
```

This will launch the tracking UI on your local machine, using port 5000 by default, but we can change the port by adding `-p <port>` or `-—port <port>` to the command.

We can also view the recorded files locally if you navigate to `mlruns/2`.

The experiment directory contains the nine child runs with the logged data, their parent run, and the `meta.yaml` file that describes the state of the experiment.

The logged data is located in the corresponding directories — artifacts, metrics, and params. Here, tags is empty because we did not log any tags.

Under `artifacts/model`, we will find the files of the model trained with the C value corresponding to the current run.

### Logging experiments automatically

Depending on what we want to accomplish with experiment tracking, setting up runs can be quite time-consuming. Fortunately, MLflow tracking has built-in functionality for automatic logging with popular data science and ML libraries: scikit-learn, Tensorflow, Pytorch, XGBoost, LightBGM, statsmodels, Spark, and Fastai.

The scikit-learn autologger defines distinct sets of data to be tracked for standalone estimators/pipelines and parameter search estimators. With the autologger, there’s no need to explicitly log metrics, parameters, or artifacts — everything is handled by MLflow automatically. But you can additionally log variables manually if you want to.

To demonstrate the autologger, we will use scikit-learn’s `GridSearchCV` algorithm on `LogisticRegression`.

```py
    # Set an experiment for automatic logging
    mlflow.set_experiment(experiment_name="auto_logging")

    # Run the autologger
    # Checking if the script is executed directly
    if __name__ == "__main__":
        # Enabling automatic logging for scikit-learn runs
        mlflow.sklearn.autolog()
        
        # Loading data
        data = datasets.load_breast_cancer()
        
        # Setting hyperparameter values to try
        params = {"C": [1, 2, 3, 4, 5, 6, 7, 8, 9]}
        
        # Instantiating LogisticRegression and GridSearchCV
        log_reg = LogisticRegression(max_iter=1000)
        grid_search = GridSearchCV(estimator=log_reg, param_grid=params)
        
        # Starting a logging run
        with mlflow.start_run() as run:
            # Fitting GridSearchCV
            grid_search.fit(X=data.data, y=data.target)
                
        # Disabling autologging
    mlflow.sklearn.autolog(disable=True)
```

We can also manually log any other data inside the `with` block — just like we did above.

After training is complete, we will be able to view the logged results in the MLflow UI.

Compared to our manual logger, the scikit-learn autologger captures more data points, including mean fit time and mean score time. With GridSearchCV, it also shows the best achieved cross-validation score and the best C value.

The directory structure for autologger runs is similar to that of manual runs. However, if you navigate to the artifacts directory of the parent run, you will notice that MLflow has also saved:

- The best fitted estimator.

- The fitted parameter search estimator.

- A CSV file with the search results.

- Plots for the training confusion matrix, the precision-recall curve, and the ROC curve.


## Packaging Projects with MLflow Projects

### What are MLflow Projects?

**MLflow Projects** is a component of MLflow that allows users to turn data science and machine learning code into packages reproducible in various environments.

As of MLflow version 1.21.0, Projects could be used to make packages for the following two environments:

- Conda. If conda is chosen as the target environment, it will be used to handle the dependencies for the packaged code.

- Docker container. MLflow supports Docker for code containerization. If your project contains non-Python code or dependencies, you should use Docker.

We will be focusing on making packages for conda environments in this post. 

NOTE: the target environment where the code is intended to be executed should have conda or Docker installed (depending on which method you use).

### Packaging Python Code for Conda Environments with Projects

In MLflow, a project is a Git repository or a local path containing your files. To create an MLflow Project, we need the following components:

- The Python scripts we want to execute.
- A conda environment YAML file to handle dependencies.
- An MLproject file to control the flow of the application.

MLflow can use any local directory or GitHub repository as a project even without MLproject or conda env files. However, configuring them gives you more fine-grained control over the project behavior. 

To package Python projects, we need to:

- Create a directory to store our Python scripts, the configuration files listed above, and files that we need to run the scripts (e.g. datasets).

- Create a conda environment YAML file.

- Create an MLproject file.

- Modify our Python scripts to accept command line arguments (optional).

### Run Packaged MLflow Projects

To run an MLflow project, navigate to the directory above it and open the terminal. 

```bash
    mlflow run experiment -e manual_logger — experiment-name manual_logging -P max_iter=1000
```

We can also run projects by calling the `mlflow.projects.run()` function from a Python script.

The Python script should be located in the directory above the project and contain code similar to the following:

```py
    # Importing MLflow
    import mlflow

    # Running a project
    mlflow.projects.run(uri="experiment",
                        entry_point="manual_logger",
                        parameters={"max_iter": 10000},
                        experiment_name="manual_logging")
```


## What are MLflow Models?

MLflow **Models** are a set of instruments that allow you to package machine learning and deep learning models into a standard format that can be reproduced in various environments.

With Models, we can package machine learning/deep learning models for deployment in a wide array of environments.

Models relies on the concept of _flavors_ which describes how packaged models should be run in the target environment. 

There are three main types of model flavors in MLflow:

- Python function. Models saved as a generic Python function can be run in any MLflow environment regardless of which ML/DL package they were built with.

- R function. An R function is the equivalent of a Python function, but in the R programming language.

- Framework-specific flavors. MLflow can also save models in a flavor that corresponds to the framework they were built in. For example, a scikit-learn model could be loaded in its native format to enable users to leverage the functionality of scikit-learn.

The contents of a packaged model will look similar to the following:

- conda.yaml: a file that lists the conda dependencies used by the model.

- input_example.json: a file that contains an example input for the model to show the expected signature of input data.

- MLmodel: a file that describes the saved flavors, the path of the model, the model’s input and output signatures, and more.

- model.pkl: the pickled model.

- requirements.txt: a file that specifies the pip dependencies of the model. By default, these are inferred from the environment where the model is packaged.

### Saving Models with MLflow Models

Here we focus on two functionalities of MLflow Models:

- **Saving** trained models in a local environment.
- **Deploying** trained models in a local environment.

Now, we need to train a scikit-learn estimator. 

We are using scikit-learn’s breast cancer dataset to fit a `LogisticRegression` estimator. 

We are splitting the data into train and test sets because we need some data after training to show how MLflow can be used to deploy models.

```py
    # import dependencies
    import mlflow
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import pandas as pd
    from mlflow.models import infer_signature, ModelSignature
    from mlflow.types import Schema, ColSpec

    # Loading data
    data = datasets.load_breast_cancer()
        
    # Splitting the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(data.data, 
                                                        data.target,
                                                        stratify=data.target)

    # Instantiating and fitting the model
    model = LogisticRegression(max_iter=1000)            
    model.fit(X=X_train, y=y_train)
```

### Providing an input signature and input examples

Once our model is trained, we can package it straight away. However, to help others get started with the packaged model, we should also supply the expected input model signature along with input examples. We can also use input signatures to enforce the input format for models loaded as Python functions.

There are two ways we can provide the input signature for our model:

- Using internal MLflow tools to infer the signature automatically.
- Specifying the input signature manually.

### Inferring the input signature automatically

To automatically infer the input signature, we can use the function `mlflow.models.infer_signature()`. 

We are going to be using a pandas.DataFrame as `model_input`. 

The main benefit of a` pandas.DataFrame` is that it can contain the column names of the input features which makes it clearer for future users what the inputs to our model are supposed to be.

```py
    # Convert train features into a DataFrame
    X_train_df = pd.DataFrame(data=X_train, columns=data.feature_names)

    # Inferthe input signature
    signature = infer_signature(model_input=X_train_df, 
                            model_output=model.predict(X_test))
    # Inspect the signature
    print(signature)
```

We can see that our inputs were inferred to be of type double, while the output was inferred to be a Tensor of type int32.

### Specifying the Input Signature Manually

If you need more control over the input signature, you can specify it manually. To do this, you need these two MLflow classes:

- ColSpec: a class that specifies the data type and name of a column in a dataset. An alternative to ColSpec is TensorSpec, which is used to represent Tensor datasets.

- Schema: a class that represents a dataset composed of ColSpec or TensorSpec objects.

We need to create separate `Schema` objects to represent our inputs and outputs.

```py
    # Example input schema for the Iris dataset
    input_schema = Schema(inputs=[
        ColSpec(type="double", name="sepal length (cm)"),
        ColSpec(type="double", name="sepal width (cm)"),
        ColSpec(type="double", name="petal length (cm)"),
        ColSpec(type="double", name="petal width (cm)"),
    ])

    # Example input schema for the Iris dataset
    output_schema = Schema(inputs=[ColSpec(type="long")])
```

We can see that `Schema` objects are lists containing `ColSpec` objects. 

ColSpec objects contain the datatypes and the names of your columns in the dataset. Y

We do not need to specify column names, but they can be beneficial if we want to make the input signature clear and easily understood.

Now, we can create schemas for the breast cancer dataset. Since this dataset has many feature columns, we will use a list comprehension:

```py
    # Create an input schema for the breast cancer dataset
    input_schema = Schema(inputs=[ColSpec(type="double", name=feature_name) 
                                  for feature_name in data.feature_names])

    # Create an output schema for the breast cancer dataset
    output_schema = Schema(inputs=[ColSpec("long")])

    # View the input schema
    print(input_schema)
    
    # View the output schema
    print("\n", output_schema)

    # Create a signature from our schemas
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
```

### Provide input examples

We can also provide an input example that matches our model signature to help users know the correct format of inputs.

We can provide input examples as a pandas.DataFrame, a numpy.ndarray, a dict, or a list.


We can use X_train_df to make sure that the input example matches the input signature, and we pick the two first data rows as an input example.

```py
    # Create an input example from our feature DataFrame
    input_example = X_train_df.iloc[:2]
```

We can also specify conda and pip dependencies to help users reproduce the project:

```py
    # Specifying a conda environment
    conda_env = {
        "channels": ["default"],
        "dependencies": ["pip"],
        "pip": ["mlflow", "cloudpickle==1.6.0"],
        "name": "mlflow-env"}

    # Specifying pip requirements
    pip_requirements = ["mlflow"]
```

### Save the model

For scikit-learn, we can use either `mlflow.sklearn.log_model()` or `mlflow.sklearn.save_model()` to package your model:

- log_model() logs the model under the current run as an artifact.
- save_model() saves the model in the specified directory relative to your project’s path.

```py
    # Save the model 
    mlflow.sklearn.save_model(sk_model=model, 
                              path="model", 
                              conda_env=conda_env, 
                              signature=signature,
                              input_example=input_example)
```

After we save the model, the model along with the environment files and the input example will be saved in the path `/model` under the current working directory.

### Log the Model as an Artifact

We could also use `log_model()`.

We start the run with `mlflow.start_run()` because we need to obtain the ID of the run to more easily load the logged model later.

```py
    # Save the model as an artifact during a run
    with mlflow.start_run() as run:
        # Obtaining the ID of this run
        run_id = run.info.run_id
        
        # Logging our model
        mlflow.sklearn.log_model(sk_model=model, 
                                 artifact_path="model", 
                                 conda_env=conda_env, 
                                 signature=signature,
                                 input_example=input_example)
```

### Load the saved models

MLflow saves scikit-learn models in two formats or flavors:

- As a Python function.
- As a native scikit-learn model.

```py
    # Path to the model saved with log_model
    model_uri_logged = "runs:/{run_id}/model"

    # Path to the model saved with save_model
    model_uri_saved = "model"

    # Load the model as a Python function
    pyfunc_model = mlflow.pyfunc.load_model(model_uri=model_uri_saved)

    # Load the model as a scikit-learn model
    sklearn_model = mlflow.sklearn.load_model(model_uri=model_uri_saved)
```

Keep in mind the following:

- To load a model that has been saved with `mlflow.sklearn.log_model()`, we can prefix the directory with `runs:/`; specify the ID of the run under which the model was saved (`{run_id}` in our case); specify the path you passed to artifact_path earlier (model). 

When we prefix the path with `runs:/`, MLflow will look for the model in the artifacts path of the indicated run.

- To load a model that has been saved with `mlflow.sklearn.save_model()`, specify the model path relative to the directory where we are running the Python script.

### Inference with the Loaded models

To perform inference with the loaded models,w e call their `predict()` methods and pass some data:

```py
    # Inference with the scikit-learn model
    sklearn_predictions = sklearn_model.predict(X_test)

    # Create a DataFrame from our test features
    X_test_df = pd.DataFrame(X_test, columns=data.feature_names)

    # Inferece with the Python function
    pyfunc_predictions = pyfunc_model.predict(X_test_df)
```

Here, `sklearn_predictions` and `pyfunc_predictions` have the same contents.

```py
    # Inspect the predictions
    print(sklearn_predictions, 
          "\n\n", 
          np.equal(pyfunc_predictions, sklearn_predictions).all())
```

With many scikit-learn models (including our sklearn_model), we can also use `predict_proba` to obtain class probabilities. 

### Serve Models with MLflow Models

We can deploy models saved with MLflow in a variety of environments, ranging from a local machine to cloud platforms like Amazon SageMaker or Microsoft Azure.

Here, we perform a local deployment of our saved model using the command line interface. 

```bash
    mlflow models serve -m path-to-model
```

### Inference with a Served Model

Now that our model is live, we can do inference with it. 

We can do this either programmatically (from a Python script) or through the command line. 

To do inference programmatically, we need to define our endpoint URL and the query function:

```py
    # Import the requests library for handling HTTP requests
    import requests

    # Declare the endpoint and payload
    url = "http://127.0.0.1:5000/invocations"

    # Definie the query function
    def query(url, payload, headers={"Content-Type": "application/json"}):
        return requests.post(url=url, 
                             data=payload, 
                             headers=headers)
```

We are passing the request header headers as a parameter to the function because we will need an easy way to change the header later.

Now, we can use the function query to do inference.

- We serialize our test_dataframe as a JSON string.
- We send a query to our model and inspect the response.

```py
    # Convert the test DataFrame to JSON with different data orientations (Records)
    payload = X_test_df.to_json(orient="records")

    # Split
    payload = X_test_df.to_json(orient="split")

    # Send POST request and obtaining the results
    response = query(url=url, 
                     payload=payload)

    # Inspect the response
    print(response.json())
```

To do inference with a CSV-serialized DataFrame, we would need to pass `{“Content-Type”: “text/csv”}` as the header in the POST request. 

```py
    # Send POST request and obtaining the results
    response = query(url=url, 
                     payload=X_test_df.to_csv(), 
                     headers={"Content-Type": "text/csv"})

    # Inspect the response
    print(response.json())
```

We can also use the command line interface to do predictions.


## Build a Python API and integrate it into a React web applicatio

Here we build a Python API and integrate it into a React web application. This application will allow us to:

- Quickly try hyperparameter combinations to train linear and logistic regression.

- Deploy best estimators and do inference.


### Overview of the Application

The React application will allow us to use some of the tracking and model deployment features of MLflow with very little code:

- Use grid search to quickly try different hyperparameter combinations for linear and logistic regression. To help you achieve this, the app uses scikit-learn’s GridSearchCV, LogisticRegression, and LinearRegression.

- Deploy the best estimators trained with GridSearchCV locally (more about MLflow deployment in PART 2).

- Do inference with the best estimators and save predictions.

The app is not meant to deploy MLflow models in production, but the app helps us get a better idea of what MLflow can do and how to use it for deployment.

We will be using Flask, Flask-RESTful, and Flask-CORS.

### Building the API

Our API will handle the following:

- Tracking experiments with MLflow.

- Training of `LogisticRegression` or `LinearRegression` with `GridSearchCV`.

- The deployment of the best estimators trained with `GridSearchCV`.

- Inference with deployed estimators.

### Building the API endpoints

Next, we are create two endpoints for our API:

- /track-experiment : used to train GridSearchCV and track experiments.

- /deploy-model : used to deploy models and do inference.



## ML Pipelines

A machine learning **pipeline** is composed of a sequence of steps that automate a machine learning **workflow**. 

Common steps in a machine learning pipeline include: data collection, data cleaning, feature engineering, model training, model evaluation.

In this article, we create a machine learning pipeline using Ploomber, Pycaret, and MLFlow for model training and batch inference [3]. 

### Ploomber

Ploomber is an open source framework used for building modularized data pipelines using a collection of python scripts, functions, or Jupyter Notebooks.

Ploomber helps to concatenate all these notebook into sequence of steps based on a user defined pipeline.yaml file. 

The figure below shows an example of a Ploomber pipeline. 

The visualize and train tasks are dependent on the clean task.

### PyCaret

PyCaret is an open-source, low-code automated machine learning (AutoML) library in python. 

PyCaret helps to simplify the model training process by automating steps such as data pre-processing, hyperparameter optimization, stacking, blending and model evaluation. 

PyCaret is integrated with MLFlow and automatically log the run’s parameters, metrics, and artifacts to the MLFlow server.

The article [3] uses the Pima Indian Diabetes Dataset from the National Institute of Diabetes and Digestive and Kidney Diseases.  

The objective of the dataset is to diagnostically predict whether or not a patient has diabetes based on certain diagnostic measurements included in the dataset. 

Several constraints were placed on the selection of these instances from a larger database. In particular, all patients here are females at least 21 years old of Pima Indian heritage. 

The datasets consists of several medical predictor variables and one binary target variable, Outcome. 

Predictor variables include the number of pregnancies the patient has had, their BMI, insulin level, age, etc. 

We split the data into two sets named diabetes_train.csv and diabetes_test.csv for developing our training pipeline and testing the serving pipeline, respectively.

The project involves the following steps:

- Define pipeline.yaml
- Create the notebooks
- Visualize the Pipeline
- Run the Training Pipeline
- Serving Pipeline
- Batch Inference



## References

[1] [Managing Machine Learning Lifecycles with MLflow](https://www.instapaper.com/read/1489714170)

[2] [MLflow Quickstart](https://mlflow.org/docs/latest/quickstart.html)

[3] [Machine Learning Pipeline with Ploomber, PyCaret and MLFlow](https://towardsdatascience.com/machine-learning-pipeline-with-ploomber-pycaret-and-mlflow-db6e76ee8a10#4a8f)

