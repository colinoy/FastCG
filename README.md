<p style="text-align: center;">
    <h1><b>FastCG</b></h1>
</p>
<!-- # FastCG -->
FastCounterfactualGenerator (or in short - FastCG) is an open‑sourced library for generating Counterfactuals for Machine Learning models in a fast and efficient way.

FastCG library is designed to be easily extensible, so that new generators can be added with ease, and while currently there is only one generator available we would love to see more people contribute and add their own generators!

We have tested the library on multiple datasets and models, and has been shown to be able to generate Counterfactuals in a fraction of the time it takes other libraries to do so WITHOUT compromising any results!

## Guiding Principles

* Weighted KNN for efficient large scale search
* PCA for reducing the dimensionality of the search space
* Binary search for optimizing the counterfactuals
* Multiprocessing for parallelizing the search and utilizing the full power of the machine

# Features

FastCG provides the following features:

- Production-ready library
- Generators
    - MultiDimentionalGenerator
- Optimizers
    - VectorizedBinarySearch
- Multiprocessing built-in
- Built in inspection utilities
- Easy to use
- Easy to extend


# Getting Started

You can get started with FastCG immediately by installing it with pip:

```bash
pip install FastCG
```

Alternatively you can install it from source:

```bash
git clone
cd FastCG
python setup.py install
```



# Usage
```python
# step one - Create a trained and ready to use model
# Important - it has to have a predict method
log_reg = LogisticRegression().fit(X_train, y_train)

# step two - import and create a generator

from FastCG.generators import ProbKnnGenerator

# step three - create a config for the generator

config = {"features_to_change": ['Current Loan Amount', 'Term', 'Credit Score', 'Annual Income', 'Monthly Debt'],
              "increase_only": ['Credit Score', 'Annual Income'], # optional
              "decrease_only": ['Monthly Debt'], # optional
              "max_features_to_change": 2,
              "ID": 'Loan ID', # optional
              "target": 'Loan Status' 
              }

# step four - create a generator
cfg_generator = ProbKnnGenerator(df,log_reg,config,target=1,condition='==') 

# step five - generate the counterfactuals

valid_ng, invalid_ng = cfg_generator.generate(X_test,n_jobs=-1,chunk_size=100)

# step six - Profit!

```


### Utilities
For your convinience we have added a few utilities that can help you with your visualization and inspection of the counterfactuals

```python
# After generating, you can inspect the valid counterfactuals 
keys = list(valid.keys())

for key in keys[:3]:
    cfg_generator.show_counterfactual(key,valid,show_plot=True)
```

This utility will provide you with a plot of the original observation, the counterfactual and the annotated difference between them
and will also print the information in a table format
<p align="center">
    <img src="https://cdn.discordapp.com/attachments/892868014159044629/1102292337675219005/image.png">
</p>

# Documentation

## Config

The config is a dictionary that contains the following keys: 

- features_to_change - a list of features that can be changed - REQUIRED
- increase_only - a list of features that can only be increased
- decrease_only - a list of features that can only be decreased
- max_features_to_change - the maximum number of features that can be changed - REQUIRED
- ID - the name of the ID column - REQUIRED
- target - the name of the target column - REQUIRED

## Generators

### ProbKnnGenerator

The ProbKnnGenerator has the following stages:

1. Pre-processing (on the entire dataset, happens when creating an instance of the generator) - 
    1. We find all the data where the model prediction matches the target (on the "green" side of the decision boundary)
    2. Reduce the dimensionality of the found data using PCA, PCA reduction is only done on the features that are "to change"
    3. Clustering the data using KMeans based on the PCA components


<p align="center">
    <img src="https://cdn.discordapp.com/attachments/892868014159044629/1102288590597259316/KNN_search.gif">
</p>

2. Generating - This step happens in the generate function
    1. Find all the observations where the model prediction DOES NOT match the target condition (lets call them instances)
    2. *For each instance:
        1. Reduce the dimensionality of the instances using PCA
        2. Find the closest two cluster centroids for each instance and select 5 members from each cluster
        3. **Using weighted KNN we find the 5 closest members to our instance based on the PCA components (essentially freezing the features that are not "to change")
        4. For each member
            1. Attempt to insert its "to change" features into the instance
            2. If the instance prediction matches the target condition (e.g - valid) we add it to the counterfactuals list
            3. We continue until we either succeed or we tried all the members


3. *Verification - This happens for each found counterfactual
    1. We check if the counterfactual is valid by checking if the model prediction matches the target condition 
    2. If it does not we mark it as invalid

<p align="center">
    <img src="https://cdn.discordapp.com/attachments/892868014159044629/1102288590010068992/BinaryVectorSearch.gif">
</p>

3. *Optimization - This happens for each valid counterfactual
    1. We use the VectorizedBinarySearch optimizer to find the optimal counterfactual
    2. The optimizer works by finding a "features to change" dimensional vector between the original observation and the found counterfactual
    3. We utilize that each vector as a start point, direction and length
    3. Using binary search we search along the length of the vector until we find the exact point of the decision boundary
    4. We then move the found point a little bit away from the decision boundary towards the counterfactual to ensure that we are not as affected by the model's error margin

***Each of the marked steps (and their nested steps) are done in Parallel and are optimized for speed and efficiency**

**We tested several variations of this KNN based search, mainly testing how would different approaches to the amount of features we base our search on. For example - given 10 PCA components do the nested logic for a KNN on every possible combination with 2 components, 3 components, 4 components and so on. We saw marginal improvements in the results but the time it took to generate counterfactuals increased exponentially.

Parameters:
- df - the DataFrame to use with entire dataset you want to use for counterfactual generation
- model - the model to use for counterfactual generation, IMPORTANT - it has to have a predict method
- config - the config dictionary
- target - the target threshold
- comparison - the comparison operator to use for the target threshold, given as a string

After creating the class it will do all the preprocessing and will be ready to generate counterfactuals.

To generate counterfactuals use the generate method:

Parameters:
- data - the data to generate counterfactuals for
- n_jobs - the number of CPU cores to use for multiprocessing, use -1 for all available CPU cores and 1 for no multiprocessing
- chunk_size - the size of the chunks to use for multiprocessing, the default is 1000
- verbose - whether to print extra information - useful for debugging, the default is 0 (no printing)

# Support


## Create a Bug Report

If you see an error message or run into an issue, please create a bug report. This effort is valued and helps all users.


## Submit a Feature Request

If you have an idea, or you're missing a capability that would make development easier and more robust, please Submit feature request.

If a similar feature request already exists, don't forget to leave a "+1".
If you add some more information such as your thoughts and vision about the feature, your comments will be embraced warmly :)


# Contributing

FastCG is an open-source project. We are committed to a fully transparent development process and highly appreciate any contributions. Whether you are helping us fix bugs, proposing new features, improving our documentation or spreading the word - we would love to have you!
