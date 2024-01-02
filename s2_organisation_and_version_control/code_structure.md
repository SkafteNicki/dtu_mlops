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
how you want your folders, files etc. to be organised from the beginning. For this course we are going to be using a
custom [MLOps template](https://github.com/SkafteNicki/mlops_template). The template is essentially a fork of the
[cookiecutter data science template](https://github.com/drivendata/cookiecutter-data-science) template that has been
used for a couple of years in the course, but specialized a bit more towards MLOps instead of general data science.

We are not going to argue that this template is better than everyother template, we are just focusing that it is a
**standardized** way of creating project structures for machine learning projects. By standardized we mean, that if two
persons are both using `cookiecutter` with the same template, the layout of their code does follow some specific rules,
making one able to faster get understand the other persons code. Code organization is therefore not only to make the
code easier for you to maintain but also for others to read and understand.

Below is seen the default code structure of cookiecutter for data science projects.

<figure markdown>
![Image](../figures/cookie_cutter.png){ width="1000" }
</figure>

What is important to keep in mind when using a template, is that it exactly is a template. By definition a template is
*guide* to make something. Therefore, not all parts of an template may be important for your project at hand. Your job
is to pick the parts from the template that is useful for organizing your machine learning project and add the parts
that are missing.

## Python projects

While the same template in principal could be used regardless of what language we where using for our machine learning
or data science application, there are certain considerations to take into account based on what language we are using.
Python is the dominant language for machine learning and data science currently, which is why we in this section is
focusing on some of the special files you will need for your Python projects.

The first file you may or may not know is the `__init__.py` file. In Python the `__init__.py` file is used to mark a
directory as a Python package. Therefore as a bare minimum, any Python package should look something like this:
package should look something like this

```txt
├── src/
│   ├── __init__.py
│   ├── file1.py
│   ├── file2.py
├── pyproject.toml
```

The second file to focus on is the `pyproject.toml`. This file is important for actually converting your code into a
Python project. Essentially, whenever you run `pip install`, `pip` is in charge of both downloading the package you want
but also in charge of *installing* it. For `pip` to be able to install a package it needs instructions on what part of
the code it should install and how to install it. This is the job of the `pyproject.toml` file.

Below we have both added a description of the structure of the `pyproject.toml` file but also `setup.py + setup.cfg`
which is the "old" way of providing project instructions regarding Python project. However, you may still encounter
a lot of projects using `setup.py + setup.cfg` so it is good to at least know about them.

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
    requires-python = ">=3.8"
    dynamic = ["dependencies"]

    [tool.setuptools.dynamic]
    dependencies = {file = ["requirements.txt"]}
    ```

    the `[build-section]` informs `pip`/`python` that to build this Python project it needs the two packages
    `setuptools` and `wheels` and that it should call the
    [setuptools.build_meta](https://setuptools.pypa.io/en/latest/build_meta.html) function to actually build the
    project. The `[project]` section essentially contains metadata regarding the package, what its called etc. if we
    ever want to publish it to [PyPI](https://pypi.org/).

    For specifying dependencies of your project you have two options. Either you specify them in a `requirements.txt`
    file and it as a dynamic field in `pyproject.toml` as shown above. Alternatively, you can add a `dependencies` field
    under the `[project]` header like this:

    ```toml
    [project]
    dependencies = [
        'torch==2.1.0',
        'matplotlib>=3.8.1'
    ]
    ```

    The improvement over `setup.py + setup.cfg` is that `pyproject.toml` also allows for metadata from other tools to
    be specified in it, essentially making sure you only need a single file for your project. For example, in the next
    [module M7 on good coding practices] you will learn about the tool `ruff` and how it can help format your code. If
    we want to configure `ruff` for our project we can do that directly in `pyproject.toml` by adding additional
    headers:

    ```toml
    [ruff]
    ruff_option = ...
    ```

    To read more about how specify `pyproject.toml` this
    [page](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/#declaring-project-metadata)
    is a good place to start.

=== "setup.py + setup.cfg"

    `setup.py` is the original way to describing how a Python package should be build. The most basic `setup.py` file
    will look like this:

    ```python
    from setuptools import setup
    from pip.req import parse_requirements
    requirements = [str(ir.req) for ir in parse_requirements("requirements.txt")]
    setup(
        name="my-package-name",
        version="0.1.0",
        author="EM",
        description="Something cool here."
        install_requires=requirements,
    )
    ```

    Essentially, the it is the exact same meta information as in `pyproject.toml`, just written directly in Python
    syntax instead of `toml`. Because there was a wish to deperate this meta information into a separate file, the
    `setup.cfg` file was created which can contain the exact same information as `setup.py` just in a declarative
    config.

    ```toml
    [metadata]
    name = my-package-name
    version = 0.1.0
    author = EM
    description = "Something cool here."
    # ...
    ```

    This non-standardized way of providing meta information regarding a package was essentially what lead to the
    creation of `pyproject.toml`.

Regardless of what way a project is configured, after creating the above files the correct way to install them would be
the same

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

after running this your code should be available to import as `from project_name import ...` like any other Python
package you use. This is the most essential you need to know about creating Python packages.

## ❔ Exercises

After having installed cookiecutter (exercise 1 and 2), the remaining exercises are intended to be used on taking the
simple CNN MNIST classifier from yesterdays exercise and force it into this structure. You are not required to fill out
every folder and file in the project structure, but try to at least follow the steps in exercises. Whenever you need to
run a file I recommend always doing this from the root directory e.g.

```bash
python <project_name>/data/make_dataset.py data/raw data/processed
python <project_name>/models/train_model.py <arguments>
etc...
```

in this way paths (for saving and loading files) are always relative to the root.

1. Install [cookiecutter](https://cookiecutter.readthedocs.io/en/stable/) framework

    ``` bash
    pip install cookiecutter
    ```

2. Start a new project using [this template](https://github.com/SkafteNicki/mlops_template), that is specialized for
    this course (1).
    { .annotate }

    1. If you feel like the template can be improve in some way, feel free to either open a issue with the proposed
        improvement or directly send a pull request to the repository 😄.

    You do this by running the cookiecutter command using the template url.

    ??? note "Flat-layout vs src-layout"

        There are two common choices on how layout your source directory. The first is called *src-layout*
        where the source code is always place in a `src/<project_name>` folder and the second is called *flat-layout*
        where the source code is place is just placed in a `<project_name>` folder. The template we are using in this
        course is using the flat-layout, but there are
        [pros and cons](https://packaging.python.org/en/latest/discussions/src-layout-vs-flat-layout/) for both.

3. After having created your new project, the first step is to also create a corresponding virtual environment and
    install any needed requirements. If you have a virtual environment from yesterday feel free to use that else create
    a new. Then install the project in that environment

    ```bash
    pip install -e .
    ```

4. Start by filling out the `<project_name>/data/make_dataset.py` file. When this file runs, it should take the raw
    data e.g. the corrupted MNIST files from yesterday which now should be located in a `data/raw` folder and process
    them into a single tensor, normalize the tensor and save this intermediate representation to the `data/processed`
    folder. By normalization here we refer to making sure the images have mean 0 and standard deviation 1.

5. This template comes with a `Makefile` that can be used to easily define common operations in a project. You do not
    have to understand the complete file but try taking a look at it. In particular the following commands may come in
    handy

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
        which enables Windows users to install packages similar to Linux system.

    In general we recommend that you add commands to the `Makefile` as you move along in the course. If you want to know
    more about how to write `Makefile`s then this is an excellent
    [video](https://youtu.be/F6DZdvbRZQQ?si=9qg-XUva-l-9Tl21).

6. Put your model file (`model.py`) into `<project_name>/models` folder together and insert the relevant code from the
    `main.py` file into the `train_model.py` file. Make sure that whenever a model is trained and it is saved, that it
    gets saved to the `models` folder (preferably in sub-folders).

7. When you run `train_model.py`, make sure that some statistics/visualizations from the trained models gets saved to
    the `reports/figures/` folder. This could be a simple `.png` of the training curve.

8. (Optional) Can you figure out a way to add a `train` command to the `Makefile` such that training can be started
    using

    ```bash
    make train
    ```

9. Fill out the newly created `<project_name>/models/predict_model.py` file, such that it takes a pre-trained model file
    and creates prediction for some data. Recommended interface is that users can give this file either a folder with
    raw images that gets loaded in or a `numpy` or `pickle` file with already loaded images e.g. something like this

    ```bash
    python <project_name>/models/predict_model.py \
        models/my_trained_model.pt \  # file containing a pretrained model
        data/example_images.npy  # file containing just 10 images for prediction
    ```

10. Fill out the file `<project_name>/visualization/visualize.py` with this (as minimum, feel free to add more
    visualizations)
    * Loads a pre-trained network
    * Extracts some intermediate representation of the data (your training set) from your cnn. This could be the
        features just before the final classification layer
    * Visualize features in a 2D space using
        [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) to do the dimensionality
        reduction.
    * Save the visualization to a file in the `reports/figures/` folder.

11. (Optional) Feel free to create more files/visualizations (what about investigating/explore the data distribution?)

12. Make sure to update the `README.md` file with a short description on how your scripts should be run

13. Finally make sure to update the `requirements.txt` file with any packages that are necessary for running your
    code (see [this set of exercises](../s1_development_environment/package_manager.md) for help)

14. (Optional) Lets say that you are not satisfied with the template I have recommended that you use, which is
    completely fine. What should you then do? You should ofcause create your own template! This is actually not that
    hard to do.

    1. Just for a starting point I would recommend that you fork either the
        [mlops template](https://github.com/SkafteNicki/mlops_template) which you have already been using or
        alternatively fork the [data science template](https://github.com/drivendata/cookiecutter-data-science)
        template.

    2. After forking the template, clone it down locally and lets start modifying it. The first step is changing
        the `cookiecutter.json` file. For the mlops template it looks like this:

        ```json
        {
            "project_name": "project_name",
            "repo_name": "{{ cookiecutter.project_name.lower().replace(' ', '_') }}",
            "author_name": "Your name (or your organization/company/team)",
            "description": "A short description of the project.",
            "python_version_number": "3.10",
            "open_source_license": ["No license file", "MIT", "BSD-3-Clause"]
        }
        ```

        simply add a new line to the json file with the name of the variable you want to add and the default value you
        want it to have.

    3. The actual template is located in the `{{ cookiecutter.project_name }}` folder. `cookiecutter` works by replacing
        everywhere that it sees `{{ cookiecutter.<variable_name> }}` with the value of the variable. Therefore, if you
        want to add a new file to the template, just add it to the `{{ cookiecutter.project_name }}` folder and make
        sure to add the `{{ cookiecutter.<variable_name> }}` where you want the variable to be replaced.

    4. After you have made the changes you want to the template, you should test it locally. Just run

        ```bash
        cookiecutter . -f --no-input
        ```

        and it should create a new folder using the default values of the `cookiecutter.json` file.

    5. Finally, make sure to push any changes you made to the template to GitHub, such that you in the future can use it
        by simply running

        ```bash
        cookiecutter https://github.com/<username>/<my_template_repo>
        ```

That ends the module on code structure and `cookiecutter`. We again want to stress the point of using `cookiecutter`
is not about following one specific template, but instead just to use any template for organizing your code. What often
happens in a team is that multiple templates are needed in different stages of the development phase or for different
product types because they share common structure, while still having some specifics. Keeping templates up-to-date then
becomes critical such that no team member is using an outdated template. If you ever end up in this situation, we highly
recommend to checkout [cruft](https://github.com/cruft/cruft) that works alongside `cookiecutter` to not only make
projects but update existing ones as template evolves. Cruft additionally also has template validation capabilities to
ensure projects match the latest version of a template.
