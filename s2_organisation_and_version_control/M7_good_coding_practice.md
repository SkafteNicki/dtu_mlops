---
layout: default
title: M7 - Good coding practice
parent: S2 - Organisation and version control
nav_order: 3
---

# Good coding practice
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

> Code is read more often than it is written. -- [Guido Van Rossum](https://gvanrossum.github.io/) (author of Python)

To understand what good coding practice are, it is important to understand what it is *not*:
* Making sure your code run fast
* Making sure that you use a specific coding paradigm (object orientated programming ect.)
* Making sure to only few dependencies

Instead good coding practices really comes down to two topics: documentation and styling.

## Documentation

Most programmers have a love-hate relationship with documentation: We absolute hate writing it ourself, but love
when someone else have actually taken time to add it to their code. There is no doubt about that well documented
code is much easier to maintain as you do not need to remember all details about the code to still maintain it.
It is key to remember that good documentation saves more time than it takes to write. 

The problem with documentation is that there is no right or wrong way to do it. You can end up doing:
* Under documentation: You document information that is clearly visible from the code and not the complex
parts that are actually hard to understand.

* Over documentation: Writing too much documentation will have the opposite effect on most people than 
what you want: there is too much to read, so people will skip it.

Here is a good rule of thump for inline comments

> “Code tells you how; Comments tell you why.” -- Jeff Atwood

### Exercises

1. Go over the most complicated file in your project. Be critical and add comments where the logic
behind the code is not easily understandable. Hint: In deep learning we often work with tensors that
change shape constantly. It is always a good idea to add comments where a tensor undergoes some reshaping.

2. Add [docstrings](https://www.python.org/dev/peps/pep-0257/) to at least two python function/methods.
You can see [here (example 5)](https://www.programiz.com/python-programming/docstrings) a good example
how to use identifiable keywords such as `Parameters`, `Args`, `Returns` which standardizes the way of
writing docstrings.

## Styling

While python already enforces some styling (e.g. code should be indented in a specific way), this is not enough
to secure that code from different users actually look like each other. Maybe even more troubling is that you
will often see that your own style of coding changes as you become more and more experienced. This kind of
difference in coding style is not that important to take care of when you are working on a personal project,
but when working multiple people together on the same project it is important to consider.

The question then remains what styling you should use. This is where [Pep8](https://www.python.org/dev/peps/pep-0008/) 
comes into play, which is the  official style guide for python. It is essentially contains what is considered "good practice" and "bad practice" when coding python. 

One way to check if your code is pep8 compliant is to use 
[flake8](https://flake8.pycqa.org/en/latest/).

### Exercises

3. Install flake8
   ```
   pip install flake8
   ```

4. run flake8 on your project
   ```
   flake8 .
   ```
   are you pep8 compliant or are you a normal mortal?

You could go and fix all the small errors that `flake8` is giving. However, in practice large projects instead relies on some kind of code formatter, that will automatically format your code for you to be pep8 compliant.
Some of the biggest are:

* [black](https://github.com/psf/black)
* [yapf](https://github.com/google/yapf)

It is important to note, that code formatting is in general not about following a specific code style, but rather that all users follow the same.

5. Install a code formatter of your own choice (I recommend black) and let it format at least one of the script in your codebase. You can also try to play around with the different formatters to find out which formatter you like the most 

One aspect not covered by `pep8` is how `import` statements in python should be organized. If you are like most
people, you place your `import` statements at the top of the file and they are ordered simply by when you needed them.
For this reason `import` statements is something we also want to take care of, but do not want to deal with ourself.

6. Install [isort](https://github.com/PyCQA/isort) the standard for sorting imports
   ``` 
   pip install isort
   ```

7. run isort on your project
   ```
   isort .
   ```
   and check how the imports were sorted.

Finally, we can also configure `black`, `isort` ect. to our specific needs. For example the recommended line length in `pep8` is 79 characters, which by many is considered very restrictive. To customize this we can create a `.flake8` file to tell `flake8` exactly how it should check our code:

```
[flake8]
exclude = venv
ignore = W503 # W503: Line break occurred before binary operator
max-line-length = 100
```

8. Add the above code snippet to a file named `.flake8` in your project. Add a line with a length longer than the standard 79 characters but below 100 and run `flake8` again to check that you get no error.

9. To make sure that your formatter still does not try to format to 79 character length, add a `pyproject.toml` file to your project where you customize the rules for the different formatters (Hint: Google is your friend). Again create a line above 79 but below 100 characters and check that it is not being formatted.

10. (Optional) Experiment further with the customization of `flake8`, `black` ect. Especially it may be worth looking into the `include` and `exclude` keywords for specifying which files should actually be formatted.

## Typing 

In addition to writing documentation and following a specific styling, in python we have a third way of improving the quality of our code: [through typing](https://docs.python.org/3/library/typing.html). Typing goes back to the earlier programming languages like `c`, `c++` ect. where data types needed to be explicit stated for variables:

```cpp
int main() {
  int x = 5 + 6;
  float y = 0.5;
  cout << "Hello World! " << x << std::endl();
}
```

This is not required by python but it can really improve the readability of code, that you can directly read from the code what the expected types of input arguments and returns are.
In python the `:` character have been reserved for type hints. Here is one example of adding typing to a function:

```python
def add2(x: int, y: int) -> int:
   return x+y
```
here we mark that both `x` and `y` are integers and using the arrow notation `->` we mark that the output type is also a integer. Assuming that we are also going to use the function for
floats we could improve the typing by specifying a `Union` of types that the function can accept:

```python
from typing import Union
def add2(x: Union[int, float], y: Union[int, float]) -> Union[int, float]:
   return x+y
```

and similar adding support for pytorch tensors:

```python
from torch import Tensor  # note it is Tensor with upper case T. This is the base class of all tensors
from typing import Union
def add2(x: Union[int, float, Tensor], y: Union[int, float, Tensor]) -> Union[int, float, Tensor]:
   return x+y
```

Finally, since this is a very generic function it also works on numpy arrays ect. we can always default to the `Any` type if we are not sure about all the specific
types that a function can take
```python
from typing import Any
def add2(x: Any, y: Any) -> Any:
   return x+y
```
However, in this case we basically is in the same case as if our function were not typed, as the type hints does not help us at all. Therefore, use `Any` only when nessesary.

### Exercises

11. We provide a file called `typing_exercise.py`. Add typing everywhere in the file. Please note that you will
need the following import:
    ```python
    from typing import Callable, Optional, Tuple, Union, List  # you will need all of them in your code
    ```
    for it to work. Hint: [here](https://mypy.readthedocs.io/en/stable/cheat_sheet_py3.html) is a good resource on typing. We also provide `typing_exercise_solution.py`, but try to solve the exercise yourself.

12. [mypy](https://mypy.readthedocs.io/en/stable/index.html) is what is called a static type checker. If you are using typing
in your code, then a static type checker can help you find common mistakes. `mypy` does not run your code, but it scans it and
checks that the types you have given are compatible. Install `mypy`

    ```bash
    pip install mypy
    ```

13. Try to run `mypy` on the `typing.py` file
    ```bash
    mypy typing_exercise.py
    ```
    If you have solved exercise 11 correctly then you should get no errors. If not `mypy` should tell you where your types are incompatible.
