# NetworkNNE

The software and data in this repository are the materials used in xxx.

## Computing Requirements

We highly recommend using `Conda` for installing requirements and implementation (an open-source package and environment management system for `Python`). For installing `Conda`, please refer to the website https://www.anaconda.com/products/distribution. To ensure that the code runs successfully, the following dependencies need to be installed:

> python 3.13 \
> numpy 2.4.1 \
> pytorch 2.8.0 \
> scipy 1.17.0 \
> scikit-learn 1.8.0 \


<span style="color:red">！This code is on developed for more user-friendly and extended to more interesting configurations.</span>


## File Structures

To implement the framework, practitioners can refer to the following key files:

* **setup.py**: Specify the econometric parameters and generate data.
* **nne_gen.py**: Construct the trainging and testing set.
* **nne_train.py**: Train the model and display results.


## Implementations

**We provide a simple implementation of our framework. For simulation experiments, practitioners should first generate their synthetic data.** 

```
python setup.py 
```

**This will generate a file with observational data and specified econometric parameter sets `training_set.pkl`**. 

Then, **the training and testing sets for neural networks can be constructed by running the commend below**. 

```
python nne_gen.py
```


The comment will then generate file containing materials required for a supervised learning task, **stored in `training_set_gen.pkl`**. 

Finally, use the following comment to train the neural networks (using data moments to predict econ parameters):

```
python nne_train.py
```


## Extensions

Further extensions to be provided


## Copyrights

This is a temporal version for a manuscript under review. The repository will be officially released and set permanently public after a formal acceptance by an academic journal for researchers in the relevant fields to implement, practically use, and conduct their extended research.

## Citations

Citatations to be provided


