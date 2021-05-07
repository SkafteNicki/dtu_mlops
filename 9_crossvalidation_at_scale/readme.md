### 9. Hyperparameter optimization
For todays exercises we will be integrating [optuna](https://optuna.readthedocs.io/en/stable/index.html) into 
our different models.

![hyperparams](../figures/hyperparameters.jpg)

### Exercises

1. Start by installing optuna:
   `pip install optuna`
   
2. Initially we will look at the `cross_validate.py` file. It implements simple K-fold cross validation of
   a random forest sklearn digits dataset (subset of MNIST). Lookover the script and try to run it.
   
3. We will now try to write the same code in optune. Please note that the script have a variable `OPTUNA=False`
   that you can use to change what part of the code should run. The three main concepts of optuna is
   
   * A trial: a single experiment
   
   * A study: a collection of trials
   
   * The objective: function to determine how "good" a trial is
   
   Lets start by writing the objective function, which we have already started in the script. For now you do
   not need to care about the `trial` argument, just assume that it contains the hyperparameters needed to
   define your random forest model. The output of the objective function should be a single number that we
   want to optimize. (HINT: did you remember to do K-fold crossvalidation inside your objective function?)
   
4. Next lets focus on the trial. Inside the `objective` function the trial should be used to suggest what
   parameters to use next. Take a look at the documentation for [trial](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html)
   or take a look at the [code examples](https://optuna.org/#code_examples) and figure out how to define
   the hyperparameter of the model.
   
5. Finally lets launch a study. It can be as simple as

   ```
   study = optuna.create_study()
   study.optimize(objective, n_trials=100)
   ```
   
   but lets play around a bit with it:
   
   5.1. By default the `.optimize` method will minimize the objective (by definition the optimum of an objective
        function is at its minimum). Is the score your objective function is returning something that should
        be minimized? If not, a simple solution is to put a `-` infront. However, look through the documentation
        on how to change the **direction**.
        
   5.2. As this example is quite simple, we can perform a full grid search. How to do this in optuna?
   

   
