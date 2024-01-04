![Logo](../figures/icons/conda.png){ align=right width="130"}

# Package managers and virtual environments

---

!!! info "Core Module"

Python is a great programming language and this is mostly due to its vast ecosystem of packages. No matter what you want
to do, there is probably a package that can get you started. Just try to remember when the last time you wrote a program
only using the [python standard library](https://docs.python.org/3/library/index.html)? Probably never. For this reason,
we need a way to install third-party packages and this is where
[package managers](https://en.wikipedia.org/wiki/Package_manager) come into play.

You have probably already used `pip` for the longest time, which is the default package manager for python. `pip` is
great for beginners but it is missing one essential feature that you will need as a developer or data scientist:
*virtual environments*. Virtual environments are an essential way to make sure that the dependencies of different
projects does not cross-contaminate each other. As an naive example, consider project A that requires `torch==1.3.0` and
project B that requires `torch==2.0`, then doing

```bash
cd project_A  # move to project A
pip install torch==1.3.0  # install old torch version
cd ../project_B  # move to project B
pip install torch==2.0  # install new torch version
cd ../project_A  # move back to project A
python main.py  # try executing main script from project A
```

will mean that even though we are executing the main script from project A's folder, it will use `torch==2.0` instead of
`torch==1.3.0` because that is the last version we installed, because in both cases `pip` will install the package into
the same environment, in this case the global environment. Instead if we did something like:

=== "Unix/macOS"

    ```bash
    cd project_A  # move to project A
    python -m venv env  # create a virtual environment in project A
    source env/bin/activate  # activate that virtual environment
    pip install torch==1.3.0  # install old torch version into the virtual environment belonging to project A
    cd ../project_B  # move to project B
    python -m venv env  # create a virtual environment in project B
    source env/bin/activate  # activate that virtual environment
    pip install torch==2.0  # install new torch version into the virtual environment belonging to project B
    cd ../project_A  # move back to project A
    source env/bin/activate  # activate the virtual environment belonging to project A
    python main.py  # succeed in executing main script from project A
    ```

=== "Windows"

    ```bash
    cd project_A  # move to project A
    python -m venv env  # create a virtual environment in project A
    .\env\Scripts\activate  # activate that virtual environment
    pip install torch==1.3.0  # install old torch version into the virtual environment belonging to project A
    cd ../project_B  # move to project B
    python -m venv env  # create a virtual environment in project B
    .\env\Scripts\activate  # activate that virtual environment
    pip install torch==2.0  # install new torch version into the virtual environment belonging to project B
    cd ../project_A  # move back to project A
    python main.py  # succeed in executing main script from project A
    ```

then we would be sure that `torch==1.3.0` is used when executing `main.py` in project A because we are using two
different virtual environments. In the above case we used the [venv module](https://docs.python.org/3/library/venv.html)
which is the build in python module for creating virtual environments. `venv+pip` is arguably a good combination
but when working on multiple projects it can quickly become a hassle to manage all the different
virtual environments yourself, remembering which python version to use, which packages to install and so on.

For this reason, a number of package managers have been created that can help you manage your virtual environments and
dependencies, with some of the most popular being:

* [conda](https://docs.conda.io/en/latest/)
* [pipenv](https://pipenv.pypa.io/en/latest/)
* [poetry](https://python-poetry.org/)
* [pipx](https://pipx.pypa.io/stable/)
* [hatch](https://hatch.pypa.io/latest/)
* [pdm](https://pdm.fming.dev/latest/)

with more being created every year ([rye](https://github.com/mitsuhiko/rye) is looking like a interesting project). This
is considered a problem in the python community, because it means that there is no standard way of managing
dependencies like in other languages like `npm` for `node.js` or `cargo` for `rust`.

<figure markdown>
![Image](../figures/standards.png){ width="700" }
<figcaption> <a href="https://xkcd.com/927/"> Image credit </a> </figcaption>
</figure>

In the course we do not care about which package manager you use, but we do care that you use one. If you are already
familiar with one package manager, then skip this exercise and continue to use that. The best recommendation that I can
give regarding package managers in general is to find one you like and then stick with it. A lot of time can be wasted
on trying to find the perfect package manager, but in the end they all do the same with some minor differences.
Checkout [this blogpost](https://alpopkes.com/posts/python/packaging_tools/) if you want a fairly up-to-date evaluation
of the different environment management and packinging tools that exist in the python ecosystem.

If you are not familiar with any package managers, then we recommend that you use `conda` and `pip` for this course. You
probably already have [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) installed
on your laptop, which is great. What conda especially does well, is that it allows you to create virtual environments
with different python versions, which can really be useful if you encounter dependencies that have not been updated in
a long time. In the course specifically we are going to recommend the following workflow

* Use `conda` to create virtual environments with specific python versions
* Use `pip` to install packages in that environment

Installing packages with `pip` inside `conda` environments have been considered a bad practice for a long time, but
since `conda>=4.6` it is considered safe to do so. The reason for this is that `conda` now has a build in compatibility
layer that makes sure that `pip` installed packages are compatible with the other packages installed in the environment.

## Python dependencies

Before we get started with the exercises, lets first talk a bit about python dependencies. One of the most common ways
to specify dependencies in the python community is through a `requirements.txt` file, which is a simple text file that
contains a list of all the packages that you want to install. The format allows you to specify the package name and
version number you want, with 7 different operators:

```txt
package1           # any version
package2 == x.y.z  # exact version
package3 >= x.y.z  # at least version x.y.z
package4 >  x.y.z  # newer than version x.y.z
package4 <= x.y.z  # at most version x.y.z
package5 <  x.y.z  # older than version x.y.z
package6 ~= x.y.z  # install version newer than x.y.z and older than x.y+1
```

In general all packages (should) follow the [semantic versioning](https://semver.org/) standard, which means that the
version number is split into three parts: `x.y.z` where `x` is the major version, `y` is the minor version and `z` is
the patch version.

The reason that we need to specify the version number is that we want to make sure that we can reproduce our code at a
later point. If we do not specify the version number, then we are at the mercy of the package maintainer to not change
the API of the package. This is especially important when working with machine learning models, as we want to make sure
that we can reproduce the exact same model at a later point.

Finally, we also need to discuss *dependency resolution*, which is the process of figuring out which packages are
compatible. This is a non-trivial problem, and there exists a lot of different algorithms for doing this. If you ever
have though that `pip` and `conda` is taking a long time to install something, then it is probably because it is trying
to figure out which packages are compatible with each other. For example, if you try to install

```bash
pip install "matplotlib >= 3.8.0" "numpy <= 1.19" --dry-run
```

then it would simply fail, because there are no combination of `matplotlib` and `numpy` under the given version
constraints that are compatible with each other. In this case we would need to relax the constraints to something like

```bash
pip install "matplotlib >= 3.8.0" "numpy <= 1.21" --dry-run
```

to make it work.

## ❔ Exercises

For hints regarding how to use `conda` you can checkout the
[cheat sheet](https://github.com/SkafteNicki/dtu_mlops/blob/main/s1_development_environment/exercise_files/conda_cheatsheet.pdf)
in the exercise folder.

1. Download and install `conda`. You are free to either install full `conda` or the much simpler version `miniconda`.
    The core difference between the two packages is that `conda` already comes with a lot of packages that you would
    normally have to install with `miniconda`. The downside is that `conda` is an much larger package which can be a
    huge disadvantage on smaller devices. Make sure that your installation is working by writing `conda help` in a
    terminal and it should show you the help message for conda. If this does not work you probably need to set some
    system variable to
    [point to the conda installation](https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10)

2. If you have successfully install conda, then you should be able to execute the `conda` command in a terminal.

    <figure markdown>
    ![Image](../figures/conda_activate.PNG){ width="700" }
    </figure>

    Conda will always tell you what environment you are currently in, indicated by the `(env_name)` in the prompt. By
    default it will always start in the `(base)` environment.

3. Try creating a new virtual environment. Make sure that it is called `my_enviroment` and that it install version
   3.11 of python. What command should you execute to do this?

    ??? warning "Use python 3.8 or higher"

        We highly recommend that you use python 3.8 or higher for this course. In general, we recommend that you use
        the second latest version of python that is available (currently python 3.11 as of writing this). This is
        because the latest version of python is often not supported by all dependencies. You can always check the status
        of different python version support [here](https://devguide.python.org/versions/).

4. Which `conda` commando gives you a list of all the environments that you have created?

5. Which `conda` commando gives you a list of the packages installed in the current environment?

    1. How do you easily export this list to a text file? Do this, and make sure you export it to
        a file called `enviroment.yaml`, as conda uses another format by default than `pip`.

    2. Inspect the file to see what is in it.

    3. The `enviroment.yaml` file you have created is one way to secure *reproducibility* between users, because
        anyone should be able to get an exact copy of you environment if they have your `enviroment.yaml` file.
        Try creating a new environment directly from you `enviroment.yaml` file and check that the packages being
        installed exactly matches what you originally had.

6. As the introduction states, it is fairly safe to use `pip` inside `conda` today. What is the corresponding `pip`
    command that gives you a list of all `pip` installed packages?  and how to you export this to `requirements.txt`
    file?

7. If you look through the requirements that both `pip` and `conda` produces then you will see that it
    is often filled with a lot more packages than what you are actually using in your project. What you are
    really interested in are the packages that you import in your code: `from package import module`.
    One way to come around this is to use the package `pipreqs`, that will automatically scan your project
    and create a requirement file specific to that.
    Lets try it out:

    1. Install `pipreqs`:

        ```bash
        pip install pipreqs
        ```

    2. Either try out `pipreqs` on one of your own projects or try it out on some other online project.
        What does the `requirements.txt` file `pipreqs` produce look like compared to the files produces
        by either `pip` or `conda`.

## 🧠 Knowledge check

1. Try executing the command

    ```bash
    pip install "pytest < 4.6" pytest-cov==2.12.1
    ```

    based on the error message you get, what would be a compatible way to install these?

    ??? success "Solution"

        As `pytess-cov==2.12.1` requires a version of `pytest` newer than `4.6`, we can simply change the command to be:

        ```bash
        pip install "pytest >= 4.6" pytest-cov==2.12.1
        ```

        but there of cause exists other solutions as well.

This ends the module on setting up virtual environments. While the methods mentioned in the exercises are great ways
to construct requirements files automatic, sometimes it is just easier to manually sit down and create the files as you
in that way secure that only the most necessary requirements are actually installed when creating a new environment.
