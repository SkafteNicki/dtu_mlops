---
layout: default
title: M16 - Unittesting
parent: S5 - Continuous Integration
nav_order: 1
---

<img style="float: right;" src="../figures/icons/pytest.png" width="130"> 

# Unit testing
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

{: .important }
> Core module

What often comes to mind for many developers, when discussing continuous integration (CI) is code testing.
CI should secure that whenever a codebase is updated it is automatically tested such that if bugs have been
introduced in the codebase it will be catched early on. If you look at the [MLOps pipeline](../figures/mlops.png), 
CI is one of cornerstones of operations part. However, it should be notes that applying CI does not magically secure 
that your code does not break. CI is only as strong as the tests that are automatically executed. CI simply structures 
and automates this.

<p align="center">
   <b> 
      “Continuous Integration doesn’t get rid of bugs, but it does make 
      them dramatically easier to find and remove.” 
      -Martin Fowler, Chief Scientist, ThoughtWorks 
   </b>
</p>

<p align="center">
  <img src="../figures/ci.png" width="600" 
  title="credits to https://devhumor.com/media/tests-won-t-fail-if-you-don-t-write-tests">
</p>

The kind of tests we are going to look at are called [unit testing](https://en.wikipedia.org/wiki/Unit_testing). Unit
testing refer to the practice of writing test that tests individual parts of your code base to test for correctness. By
unit you can therefore think a function, module or in general any object. By writing tests in this way it should be 
very easy to isolate which part of the code that broke after an update to the code base. Another way to test your code
base would be through [integration testing](https://en.wikipedia.org/wiki/Integration_testing) which is equally
important but we are not going to focus on it in this course.

## Pytest

Before we can begin to automatize testing of our code base we of cause need to write the tests first. It is both a hard 
and tedious task to do but arguable the most important aspects of CI. Python offers a couple of different libraries
for writing tests. We are going to use `pytest`. 

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

   1. Start by creating a `tests/__init__.py` file and fill in the following:
      ```python
      import os
      _TEST_ROOT = os.path.dirname(__file__)  # root of test folder
      _PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
      _PATH_DATA = os.path.join(_PROJECT_ROOT, "Data")  # root of data
      ```
      these can help you refer to your data files during testing. For example, in another test
      file I could write 
      ```python
      from tests import _PATH_DATA
      ```
      which then contains the root path to my data.

   2. Data testing: In a file called `tests/test_data.py` implement at least a
      test that checks that data gets correctly loaded. By this we mean that you should check
      ```python
      dataset = MNIST(...)
      assert len(dataset) == N_train for training and N_test for test
      assert that each datapoint has shape [1,28,28] or [728] depending on how you choose to format
      assert that all labels are represented
      ```
      where `N_train` should be either 25000 or 40000 depending on if you are just the first
      subset of the corrupted Mnist data or also including the second subset. `N_test` should 
      be 5000.

   3. Model testing: In a file called `tests/test_model.py` implement at least a test that
      checks for a given input with shape *X* that the output of the model have shape *Y*
        
   4. Training testing: In a file called `tests/test_training.py` implement at least one
      test that asserts something about your training script. You are here given free hands on 
      what should be tested, but try to test something the risk being broken when developing the code.
        
   5. Good code raises errors and give out warnings in appropriate places. This is often in  
      the case of some invalid combination of input to your script. For example, you model 
      could check for the size of the input given to it (see code below) to make sure it corresponds 
      to what you are expecting. Not implementing such errors would still result in Pytorch failing 
      at a later point due to shape errors, however these custom errors will probably make more sense 
      to the end user. Implement at least one raised error or warning somewhere in your code and 
      use either `pytest.raises` or `pytest.warns` to check that they are correctly raised/warned.
      As inspiration, the following implements `ValueError` in code belonging to the model:
      ```python
      # src/models/model.py
      def forward(self, x: Tensor):
         if x.ndim != 4:
            raise ValueError('Expected input to a 4D tensor')
         if x.shape[1] != 1 or x.shape[2] != 28 or x.shape[3] != 28:
            raise ValueError('Expected each sample to have shape [1, 28, 28]')
      ```
      which would be captured by a test looking something like this:
      ```python
      # tests/test_model.py
      def test_error_on_wrong_shape():
         with pytest.raises(ValueError, match='Expected input to a 4D tensor')
            model(torch.randn(1,2,3))
      ```

   6. A test is only as good as the error message it gives, and by default `assert`
      will only report that the check failed. However, we can help our self and others by adding 
      strings after `assert` like
      ```python
      assert len(train_dataset) == N_train, "Dataset did not have the correct number of samples"
      ```
      Add such comments to the assert statements you just did.

   7. The tests that involve checking anything that have to do with our data, will of cause fail
      if the data is not present. To future proof our code, we can take advantage of the
      `pytest.mark.skipif` decorator. Use this decorator to skip your data tests if the corresponding
      data files does not exist. It should look something like this
      ```python
      import os.path
      @pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
      def test_something_about_data():
         ...
      ```
      You can read more about skipping tests [here](https://docs.pytest.org/en/latest/how-to/skipping.html)

5. After writing the different tests, make sure that they are passing locally.

6. We often want to check a function/module for various input arguments. In this case you could 
   write the same test over and over again for the different input, but `pytest` also have build 
   in support for this with the use of the 
   [pytest.mark.parametrize decorator](https://docs.pytest.org/en/6.2.x/parametrize.html). 
   Implement a parametrize test and make sure that it runs for different input. 

7. There really do not exist any way of measuring how good the test you have written are. However, 
   what we can measure is the *code coverage*. Code coverage refers to the percentage of your 
   codebase that actually gets run when all your tests are executed. Having a high coverage
   at least means that all your code will run when executed.

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
      which will give you the percentage of cover in each of your files. 
      You can also write
      ```bash
      coverage report -m
      ```
      to get the exact lines that was missed by your tests.

   4. Finally, try to increase the coverage by writing a new test that runs some
      of the lines in your codebase that is not covered yet.

   5. Often `coverage` reports the code coverage on files that we actually do not want
      to get a code coverage for. Figure out how to configure `coverage` to exclude
      some files.

That covers the basic of writing unittests for python code. We want to note that `pytest` of cause is not the only
framework for doing this. Python actually have an build in framework called 
[unittest](https://docs.python.org/3/library/unittest.html) for doing this also (but `pytest` offers a bit more 
features). Another open-source framework that you could choose to checkout is
[hypothesis](https://github.com/HypothesisWorks/hypothesis) that can really help catch errors in corner cases of your
code. In addition to writing unittests it is also highly recommended to test code that you include in your
docstring belonging to your functions and modulus to make sure that any code there is in your documentation is also
correct. For such testing we can highly recommend using pythons build-in framework 
[doctest](https://docs.python.org/3/library/doctest.html).