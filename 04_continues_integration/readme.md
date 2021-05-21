# 4. Continues-Integration

Continues integration (CI) is a development practise that makes sure that updates to code 
are automatically tested such that it does not break existing code. When we look at MLOps,
CI belongs to the operation part. 

It should be notes that applying CI does not magically secure that your code does not break.
CI is only as strong as the tests that are automatically executed. CI simply structures and
automates this.

<p align="center">
<b> “Continuous Integration doesn’t get rid of bugs, but it does make them dramatically easier to find and remove.” -Martin Fowler, Chief Scientist, ThoughtWorks </b>
</p>

<p align="center">
  <img src="../figures/ci.png" width="600" title="hover text">
</p>

(All credit to [link](https://devhumor.com/media/tests-won-t-fail-if-you-don-t-write-tests))


## Pytest

The first part of continues integration is writing tests. It is both a hard and tidious task to do but
arguable the most important aspects of continues integration. Python offers a couple of different libaries
for writing tests. We are going to use `pytest`

### Exercises

The following exercises should be applyed to your MNIST reposatory

1. The first part of doing CI is writing the unit tests. We do not expect you to cover every part
   of the code you have developed but try to atleast write tests that cover two files. Start by
   creating a `tests` folder.

2. Read the [getting started guide](https://docs.pytest.org/en/6.2.x/getting-started.html) for pytest
   which is the testing framework that we are going to use
   
3. Install pytest:

   ```
   pip install pytest
   ```
   
4. Write some tests. Below are some guidelines on some tests that should be implemented, but
   you are ofcause free to implement more tests. You can at any point check if your tests are
   passing by typing in a terminal
   
   ```
   pytest tests/
   ```

   4.1. Data testing: In a file called `tests/test_data.py` implement atleast a test that
        checks that data gets correctly loaded. By this we mean that you should check
        ```
        dataset = MNIST(...)
        assert len(dataset) == 60000 for training and 10000 for test
        assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
        assert that all labels are represented
        ```

   4.2. Model testing: In a file called `tests/test_model.py` implement atleast a test that
        checks for a given input with shape *X* that the output of the model have shape *Y*
        
   4.3. Training testing: In a file called `tests/test_training.py` implement atleast one test
        that asserts something about your training script. You are here given free hands on what
        should be tested, but try to test something the risk being broken when developing the code.
        
   4.4. Good code raises errors and give out warnings in appropiate places. This is often in the case of some
        invalid combination of input to your script. For example, you model could check for the size of the input
        given to it:
        ```
        def forward(self, x: Tensor):
            if x.ndim != 4:
                raise ValueError('Expected input to a 4D tensor')
            if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3]
                raise ValueError('Expected each sample to have shape [1, 28, 28]')
        ```
        Your code would probably still fail with shape errors by PyTorch without these checks but the errors 
        would make much less sense to a enduser. Implement atleast one raised error or warning somewhere in
        your code and use either `pytest.raises` or `pytest.warns` to check that they are correctly raised/warned.

5. Finally, make sure that all your tests pass locally

6. (Optional). We often want to check a function/module for various input arguments. In this case you could
   write the same test over and over again for the different input, but `pytest` also have build in support
   for this with the use of the [pytest.mark.parametrize decorator](https://docs.pytest.org/en/6.2.x/parametrize.html).
   Implement a parametrize test and make sure that it runs for different input.

## Github actions
Github actions are the CI solution that github provides. Each of your reposatories gets 2,000 minutes of free 
testing per month which should be more than enough for the scope of this course (and probably all personal 
projects you do). Getting Github actions setup in a reposatory may seem complicated at first, but workflow
files that you create for one reposatory can more or less be reused for any other reposatory that you have.

1. Start by creating a `.github` folder in the base of your reposatory. Add a subfolder to that called `workflows`

2. Go over this [page](https://docs.github.com/en/actions/guides/building-and-testing-python) that explains
   how to do automated testing of python code in github actions. You do not have to understand everything,
   but atleast get a feeling of what a workflow file should look like.
   
3. We have provided a workflow file called `tests.yml` that should do run your tests for you. Place this file
   in the `.github/workflows/` folder. The workflow file consist of three steps
   
   * First a python enviroment is setup (in this case python 3.8)
   
   * Next all dependencies required to run the test are installed
   
   * Finally, `pytest` is called and test will be run

4. For the script to work you need to define the `requirements.txt` and `requirements_tests.txt`. The first
   file should contain all packages required to run your code. The second file is all *additional*  packages
   required to run the tests. In your simple case it may very well be that the second file is empty, however
   sometimes additional packages are used for testing that are not strictly required for the scripts to run.
   
5. Finally, try pushing the changes to your reposatory. Hopefully your tests should just start, and you will
   after sometime see a green checkermark next to hash of the commit. Also try to checkout the *Actions*  tap
   where you can see the history of actions run.

![action](../figures/action.PNG)

6. (Optional) Normally we develop code one operating system and just hope that it will work on other operating
   systems. However, CI enables us to automatically test on other systems than ourself.
   
   6.1 The provided `tests.yml` only runs on one operating system. Which one?
   
   6.2 Alter the file (or write a new) that executes the test on the two other main operating systems that exist

## Auto linter

In part 2 of the course you where introduced to a couple of good coding practises such as being consistent
with how your packages are sorted and that your code follows certain standards. In this set of exercises we
will setup workflows that will automatically test for this. 

1. Create a new workflow file called `isort.yml`, that implements the following three steps

   * Setup python enviroment
   
   * Installs `isort`
   
   * Runs `isort` on the reposatory
   
   (HINT: You should be able to just change the last steps of the `tests.yml` workflow file)
   
2. Create a new workflow file called `flake8.yml`, that implements the following three steps

   * Setup python enviroment
   
   * Installs `flake8`
   
   * Runs `flake8` on the reposatory
   
   (HINT: You should be able to just change the last steps of the `tests.yml` workflow file)
