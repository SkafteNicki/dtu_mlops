![Logo](../figures/icons/cookiecutter.png){ align=right width="130"}

# Code organization

---

!!! info "Core Module"

With a basic understanding of version control, it is now time to really begin filling up our code repository. However,
the question then remains how to organize our code? As developers we tend to not think about code organization that
much. It is instead something that just dynamically is being created as we may need it. However, maybe we should spend
some time initially getting organized with the chance of this making our code easier to develop and maintain in the
long run. If we do not spend time organizing our code, we may end up with a mess of code that is hard to understand
or maintain

!!! quote "Big ball of Mud"
    *A Big Ball of Mud is a haphazardly structured, sprawling, sloppy, duct-tape-and-baling-wire, spaghetti-code*
    *jungle. These systems show unmistakable signs of unregulated growth, and repeated, expedient repair. Information*
    *is shared promiscuously among distant elements of the system, often to the point where nearly all the important*
    *information becomes global or duplicated.* <br>
    *The overall structure of the system may never have been well defined.* <br>
    *If it was, it may have eroded beyond recognition. Programmers with a shred of architectural sensibility shun these*
    *quagmires. Only those who are unconcerned about architecture, and, perhaps, are comfortable with the inertia of*
    *the day-to-day chore of patching the holes in these failing dikes, are content to work on such systems.*
    <br> <br>
    Brian Foote and Joseph Yoder, Big Ball of Mud. Fourth Conference on Patterns Languages of Programs
    (PLoP '97/EuroPLoP '97) Monticello, Illinois, September 1997

We are here going to focus on the organization of data science projects and machine learning projects. The core
difference this kind of projects introduces compared to more traditional systems, is *data*. The key to modern machine
learning is without a doubt the vast amounts of data that we have access to today. It is therefore not unreasonable that
data should influence our choice of code structure. If we had another kind of application, then the layout of our
codebase should probably be different.

## Cookiecutter

We are in this course going to use the tool [cookiecutter](https://cookiecutter.readthedocs.io/en/latest/README.html), 
which is tool for creating projects from *project templates*. A project template is in short ust a overall structure of 
how you want your folders, files etc. to be organised from the beginning. In particular for this course we are going to 
be using the [cookiecutter data science template](https://github.com/drivendata/cookiecutter-data-science). We are not 
going to argue that this template is better than everyother template, we are just focusing that it is a **standardized** 
way of creating project structures for data science projects. By standardized we mean, that if two persons are both 
using `cookiecutter` with the same template, the layout of their code does follow some specific rules, making one able 
to faster get understand the other persons code. Code organization is therefore not only to make the code easier for 
you to maintain but also for others to read and understand.

Below is seen the default code structure of cookiecutter for data science projects.

<figure markdown>
  ![Image](../figures/cookie_cutter.png){ width="1000" }
  <figcaption> <a href="https://github.com/drivendata/cookiecutter-data-science"> Image credit </a> </figcaption>
</figure>

What is important to keep in mind when using a template, is that it exactly is a template. By
definition a template is *guide* to make something. Therefore, not all parts of an template may be important for your
project at hand. Your job is to pick the parts from the template that is useful for organizing your data science.
project.

## Making a python package

Before we get started with the exercises, there is another topic that is important to discuss and that is how 
to create python packages. Python is the dominant language for machine learning and data science currently. The
reason we want to have some itu


The hole idea



Whenever you run `pip install`, `pip` is in charge of both downloading the package you want but also
in charge of *installing* it. For `pip` to be able to install a package it needs instructions on what part of the code
it should install. The first file you should have encount

In Python the `__init__.py` file is used to mark a directory as a Python package. Therefore as a bare minimum a python
package should look something like this

```txt
├── src
│   ├── __init__.py
│   ├── file1.py
│   ├── file2.py 
├── pyproject.toml
```

We are not going to create just a selection of script and hope that they talk to each others.

=== "pyproject.toml"

    `pyproject.toml` is the new standardized way of describing project metadata in a declaratively way, introduced in
    [PEP 621](https://peps.python.org/pep-0621/). It is written [toml format](https://toml.io/en/) which is easy to
    read. At the very least your `pyproject.toml` file should include the `[build-system]` and `[project]` sections:

    ```toml
    [build-system]
    requires = ["setuptools", "wheel"]
    build-backend = "setuptools.build_meta"

    [project]
    name = "my-package-name"
    version = "0.1.0"
    authors = [{name = "EM", email = "me@em.com"}]
    description = "Something cool here."
    ```

    the `[build-section]` informs `pip`/`python` that to build this python project it needs the two packages
    `setuptools` and `wheels` and that it should call the `setuptools.build_meta` function to actually build the
    project. The `[project]` section essentially tells what the 


    if you want to be compatible with the old way of doing package in python, you can simply add a file called
    `setup.py` that includes the following code

    ```python
    from setuptools import setup
    setup()
    ```

=== "setup.py + setup.cfg"

    `setup.py` is the original way to describe

    ```python
    from setuptools import setup

    setup(
        name="my-package-name",
        version="0.1.0",
        author="EM",
        description="Something cool here."
        # ...
    )
    ```

    ```toml
    [metadata]
    name = my-package-name
    version = 0.1.0
    author = EM
    description = "Something cool here."
    # ...
    ```

https://setuptools.pypa.io/en/latest/build_meta.html


Regardless of what way you one chooses to go around the question of build systems and meta data, after creating the
above files the correct way to install them would be the same

```bash
pip install .
# or in developer mode
pip install -e . # (1)!
```

1. :man_raising_hand: The `-e` is short for `--editable` mode also called
    [developer mode](https://setuptools.pypa.io/en/latest/userguide/development_mode.html). Since we will continuously
    iterating on our package this is the preferred way to install our package, because that means that we do not have
    to run `pip install` every time we make a change. Essentially, in developer mode changes in the Python source code
    can immediately take place without requiring a new installation.


### ❔ Exercises

After having installed cookiecutter (exercise 1 and 2), the remaining exercises are intended to be used on taking the
simple CNN MNIST classifier from yesterdays exercise and force it into this structure. You are not required to fill out
every folder and file in the project structure, but try to at least follow the steps in exercises. Whenever you need to
run a file I recommend always doing this from the root directory e.g.

```bash
python src/data/make_dataset.py data/raw data/processed
python src/models/train_model.py <arguments>
etc...
```

in this way paths (for saving and loading files) are always relative to the root.

1. Start by reading [this page](https://drivendata.github.io/cookiecutter-data-science/), as it will give you insight
    to why standardized code organization is important.

2. Install [cookie cutter for data science](https://github.com/drivendata/cookiecutter-data-science)

    ``` bash
    # install using the terminal
    pip install cookiecutter
    ```

3. Take a look at the webpage to see how you start a new project. We recommend using `v2` of cookiecutter.

4. After having created your project we are going to install it as a package in our conda environment. Either run

    ```bash
    # install in a terminal in your conda env
    pip install -e .
    # or
    conda develop .
    ```

    In addition you may need to run

    ```bash
    pip install -r requirements.txt
    ```

    to install additional packages required by `cookie-cutter`.

5. Start by filling out the `src/data/make_dataset.py` file. When this file runs, it should take the raw data files in
    `data/raw` (the files that we have provided) process them into a single tensor, normalize the tensor and save this
    intermediate representation to the `data/processed` folder. By normalization here we refer to making sure the
    images have mean 0 and standard deviation 1.

6. Every `cookie-cutter` project comes with a build in `Makefile` that can be used to easily define common operations in
    a project. You do not have to understand the complete file by try taking a look at it. In particular the following
    commands may come in handy

    ```bash
    make data  # runs the make_dataset.py file, try it!
    make clean  # clean __pycache__ files
    make requirements  # install everything in the requirements.py file
    ```

    ??? note "Windows users"

        `make` is a GNU build tool that is by default not available on Windows. There are two recommended ways to get
        it running on Windows. The first is leveraging
        [linux subsystem](https://docs.microsoft.com/en-us/windows/wsl/install-win10) for Windows which you maybe have
        already installed. The second option is utilizing the [chocolatey](https://chocolatey.org/) package manager,
        which enables Windows users to install packages similar to Linux system. The second option is running

7. Put your model file (`model.py`) into `src/models` folder together and insert the relevant code from the `main.py`
    file into the `train_model.py` file. Make sure that whenever a model is trained and it is saved, that it gets saved
    to the `models` folder (preferably in sub-folders).

8. When you run `train_model.py`, make sure that some statistics/visualizations from the trained models gets saved to
    the `reports/figures/` folder. This could be a simple `.png` of the training curve.

9. (Optional) Can you figure out a way to add a `train` command to the `Makefile` such that training can be started
    using

    ```bash
    make train
    ```

10. Fill out the newly created `src/models/predict_model.py` file, such that it takes a pre-trained model file and
    creates prediction for some data. Recommended interface is that users can give this file either a folder with raw
    images that gets loaded in or a `numpy` or `pickle` file with already loaded images e.g. something like this

    ```bash
    python src/models/predict_model.py \
        models/my_trained_model.pt \  # file containing a pretrained model
        data/example_images.npy  # file containing just 10 images for prediction
    ```

11. Fill out the file `src/visualization/visualize.py` with this (as minimum, feel free to add more visualizations)
    * Loads a pre-trained network
    * Extracts some intermediate representation of the data (your training set) from your cnn. This could be the
        features just before the final classification layer
    * Visualize features in a 2D space using
        [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) to do the dimensionality
        reduction.
    * Save the visualization to a file in the `reports/figures/` folder.

12. (Optional) Feel free to create more files/visualizations (what about investigating/explore the data distribution?)

13. Make sure to update the `README.md` file with a short description on how your scripts should be run

14. Finally make sure to update the `requirements.txt` file with any packages that are necessary for running your
    code (see [this set of exercises](../s1_development_environment/package_manager.md) for help)

## 🧠 Knowledge check

??? question "Knowledge question 1"

    If tensor `a` has shape `[N, d]` and tensor `b` has shape `[M, d]` how can we calculate the pairwise distance
    between rows in `a` and `b` without using a for loop?

    ??? success "Solution"

        We can take advantage of [broadcasting](https://pytorch.org/docs/stable/notes/broadcasting.html) to do this




That ends the module on code structure and `cookiecutter`. We again want to stress the point that `cookiecutter` is
just one template for organizing your code. What often happens in a team is that multiple templates are needed in
different stages of the development phase or for different product types because they share common structure, while
still having some specifics. Keeping templates up-to-date then becomes critical such that no team member is using an
outdated template. If you ever end up in this situation, we highly recommend to checkout
[cruft](https://github.com/cruft/cruft) that works alongside `cookiecutter` to not only make projects but update
existing ones as template evolves. Cruft additionally also has template validation capabilities to ensure projects
match the latest version of a template.
