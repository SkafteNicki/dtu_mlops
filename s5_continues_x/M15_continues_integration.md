---
layout: default
title: M15 - Continues Integration
parent: S5 - Continuous X
nav_order: 1
---

# Continues Integration
{: .no_toc }

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---

Continues integration (CI) is a development practice that makes sure that updates to code are automatically tested such that it does not break existing code. If you look at the MLOps pipeline, CI is one of cornerstones of operations part. However, it should be notes that applying CI does not magically secure that your code does not break. CI is only as strong as the tests that are automatically executed. CI simply structures and automates this.

<p align="center">
<b> “Continuous Integration doesn’t get rid of bugs, but it does make them dramatically easier to find and remove.” -Martin Fowler, Chief Scientist, ThoughtWorks </b>
</p>

<p align="center">
  <img src="../figures/ci.png" width="600" title="credits to https://devhumor.com/media/tests-won-t-fail-if-you-don-t-write-tests">
</p>


## Pytest

The first part of continues integration is writing tests. It is both a hard and tedious task to do but
arguable the most important aspects of continues integration. Python offers a couple of different libraries
for writing tests. We are going to use `pytest`

### Exercises

The following exercises should be applied to your MNIST repository

1. The first part of doing CI is writing the unit tests. We do not expect you to cover every part
   of the code you have developed but try to at least write tests that cover two files. Start by
   creating a `tests` folder.

2. Read the [getting started guide](https://docs.pytest.org/en/6.2.x/getting-started.html) for pytest
   which is the testing framework that we are going to use
   
3. Install pytest:

   ```
   pip install pytest
   ```
   
4. Write some tests. Below are some guidelines on some tests that should be implemented, but
   you are of course free to implement more tests. You can at any point check if your tests are
   passing by typing in a terminal
   
   ```
   pytest tests/
   ```

   1. Data testing: In a file called `tests/test_data.py` implement at least a
      test that checks that data gets correctly loaded. By this we mean that you should check

      ```python
      dataset = MNIST(...)
      assert len(dataset) == 60000 for training and 10000 for test
      assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
      assert that all labels are represented
      ```

   2. Model testing: In a file called `tests/test_model.py` implement at least a test that
      checks for a given input with shape *X* that the output of the model have shape *Y*
        
   3. Training testing: In a file called `tests/test_training.py` implement at least one
      test that asserts something about your training script. You are here given free hands on what should be tested, but try to test something the risk being broken when developing the code.
        
   4. Good code raises errors and give out warnings in appropriate places. This is often in  
      the case of some  invalid combination of input to your script. For example, you model could check for the size of the input given to it (see code below) to make sure it corresponds to what you are expecting. Not implementing  such errors would still result in Pytorch failing at a later point due to shape errors, however these custom errors will probably make more sense to the end user. Implement at least one raised error or warning somewhere in your code and use either `pytest.raises` or `pytest.warns` to check that they are correctly raised/warned.

      ```python
      def forward(self, x: Tensor):
         if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
         if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3]
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
      ```

5. Finally, make sure that all your tests pass locally

6. We often want to check a function/module for various input arguments. In this case you could write the same test over and over again for the different input, but `pytest` also have build in support for this with the use of the [pytest.mark.parametrize decorator](https://docs.pytest.org/en/6.2.x/parametrize.html). Implement a parametrize test and make sure that it runs for different input. (HINT: you can just edit one of the test )

7. There really do not exist any way of measuring how good the test you have written are. However, what we can measure is the *code coverage*. Code coverage refers to the percentage of your codebase that actually gets run when all your tests are executed. Having a high coverage at least means that all your code will run when executed.

   1. Install coverage
      ```bash
      pip install coverage
      ```

   2. Instead of running your tests directly with `pytest`, now do
      ```bash
      coverage run -m pytest tests/
      ```

   3. To get a simple coverage report simply type
      ```bash
      coverage report
      ```
      which will give you the percentage of cover in each of your files. You can also
      write
      ```bash
      coverage report -m
      ```
      to get the exact lines that was missed by your tests.

   4. Finally, try to increase the coverage by writing a new test that runs some
      of the lines in your codebase that is not covered yet.

## Github actions
Github actions are the CI solution that github provides. Each of your repositories gets 2,000 minutes of free testing per month which should be more than enough for the scope of this course (and probably all personal projects you do). Getting Github actions setup in a repository may seem complicated at first, but workflow files that you create for one repository can more or less be reused for any other repository that you have.

Lets take a look at how a github workflow file is organised:

* Initially we start by giving the workflow a `name`
* Next we specify on what events the workflow should be triggered. This includes both the action (pull request, push ect) and on what branches is should activate
* Next we list the jobs that we want to do. Jobs are by default executed in parallel but can also be dependent on each other
* In the `runs-on` we can specify which operation system we want the workflow to run on. We also have the possibility to specify multiple.
* Finally we have the `steps`. This is where we specify the actual commands that should be run when the workflow is executed.

<p align="center">
  <img src="../figures/actions.png" width="1000" title="credits to https://madewithml.com/courses/mlops/cicd/#github-actions">
</p>

### Exercises

1. Start by creating a `.github` folder in the base of your repository. Add a subfolder to that called `workflows`

2. Go over this [page](https://docs.github.com/en/actions/guides/building-and-testing-python) that explains how to do automated testing of python code in github actions. You do not have to understand everything, but at least get a feeling of what a workflow file should look like.
   
3. We have provided a workflow file called `tests.yml` that should do run your tests for you. Place this file in the `.github/workflows/` folder. The workflow file consist of three steps
   
   * First a python environment is setup (in this case python 3.8)
   
   * Next all dependencies required to run the test are installed
   
   * Finally, `pytest` is called and test will be run

4. For the script to work you need to define the `requirements.txt` and `requirements_tests.txt`. The first file should contain all packages required to run your code. The second file is all *additional*  packages required to run the tests. In your simple case it may very well be that the second file is empty, however sometimes additional packages are used for testing that are not strictly required for the scripts to run.
   
5. Finally, try pushing the changes to your repository. Hopefully your tests should just start, and you will after sometime see a green check mark next to hash of the commit. Also try to checkout the *Actions*  tap where you can see the history of actions run.

![action](../figures/action.PNG)

6. (Optional) Normally we develop code one operating system and just hope that it will work on other operating
   systems. However, CI enables us to automatically test on other systems than ourself.
   
   1. The provided `tests.yml` only runs on one operating system. Which one?
   
   2. Alter the file (or write a new) that executes the test on the two other main operating systems that exist

7. As the workflow is currently setup, github actions will destroy every downloaded package when the workflow has been executed. To improve this we can take advantage of `caching`:

   1. Figure out how to implement `caching` in your workflow file

   2. Measure how long your workflow takes before and after adding `caching` to your workflow

## Auto linter

In [this module](../s2_organisation_and_version_control/M7_good_coding_practice.md) of the course you where introduced to a couple of good coding practices such as being consistent with how your python packages are sorted and that your code follows certain standards. In this set of exercises we will setup github workflows that will automatically test for this. 

1. Create a new workflow file called `isort.yml`, that implements the following three steps

   * Setup python environment
   
   * Installs `isort`
   
   * Runs `isort` on the repository
   
   (HINT: You should be able to just change the last steps of the `tests.yml` workflow file)
   
2. Create a new workflow file called `flake8.yml`, that implements the following three steps

   * Setup python environment
   
   * Installs `flake8`
   
   * Runs `flake8` on the repository
   
   (HINT: You should be able to just change the last steps of the `tests.yml` workflow file)
