![Logo](../figures/icons/conda.png){ align=right width="130"}

# Package managers and virtual environments

---

!!! info "Core Module"

Python is a great programming language and this is mostly due to its vast ecosystem of packages. No matter what you want
to do, there is probably a package that can get you started. Just try to remember the last time you wrote a program
only using the [Python standard library](https://docs.python.org/3/library/index.html). Probably never. For this reason,
we need a way to install third-party packages and this is where
[package managers](https://en.wikipedia.org/wiki/Package_manager) come into play.

You have probably already used `pip` for the longest time, which is the default package manager for Python. `pip` is
great for beginners but it is missing one essential feature that you will need as a developer or data scientist:
*virtual environments*. Virtual environments are an essential way to make sure that the dependencies of different
projects do not cross-contaminate each other. As a naive example, consider project A that requires `torch==1.3.0` and
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
`torch==1.3.0` because that is the last version we installed. In both cases, `pip` will install the package into
the same environment, in this case, the global environment. Instead, if we did something like:

=== "Unix/macOS"

    ```bash
    cd project_A  # move to project A
    python -m venv env  # create a virtual environment in project A
    source env/bin/activate  # activate that virtual environment
    pip install torch==1.3.0  # Install the old torch version into the virtual environment belonging to project A
    cd ../project_B  # move to project B
    python -m venv env  # create a virtual environment in project B
    source env/bin/activate  # activate that virtual environment
    pip install torch==2.0  # Install new torch version into the virtual environment belonging to project B
    cd ../project_A  # Move back to project A
    source env/bin/activate  # Activate the virtual environment belonging to project A
    python main.py  # Succeed in executing the main script from project A
    ```

=== "Windows"

    ```bash
    cd project_A  # Move to project A
    python -m venv env  # Create a virtual environment in project A
    .\env\Scripts\activate  # Activate that virtual environment
    pip install torch==1.3.0  # Install the old torch version into the virtual environment belonging to project A
    cd ../project_B  # Move to project B
    python -m venv env  # Create a virtual environment in project B
    .\env\Scripts\activate  # Activate that virtual environment
    pip install torch==2.0  # Install new torch version into the virtual environment belonging to project B
    cd ../project_A  # Move back to project A
    .\env\Scripts\activate  # Activate the virtual environment belonging to project A
    python main.py  # Succeed in executing the main script from project A
    ```

then we would be sure that `torch==1.3.0` is used when executing `main.py` in project A because we are using two
different virtual environments. In the above case, we used the [venv module](https://docs.python.org/3/library/venv.html)
which is the built-in Python module for creating virtual environments. `venv+pip` is arguably a good combination
but when working on multiple projects it can quickly become a hassle to manage all the different
virtual environments yourself, remembering which Python version to use, which packages to install and so on.

For this reason, a number of package managers have been created that can help you manage your virtual environments and
dependencies, with some of the most popular being:

```python exec="1"
# this code is being executed at build time to get the latest number of stars
import requests

def get_github_stars(owner_repo):
    url = f"https://api.github.com/repos/{owner_repo}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data.get("stargazers_count", 0)
    else:
        return None

table =  "| 🌟 Framework | 📄 Docs | 📂 Repository | ⭐ GitHub Stars |\n"
table += "|--------------|---------|---------------|----------------|\n"

data = [
    ("Conda", "https://docs.conda.io/en/latest/", "conda/conda"),
    ("Pipenv", "https://pipenv.pypa.io/en/latest/", "pypa/pipenv"),
    ("Poetry", "https://python-poetry.org/", "python-poetry/poetry"),
    ("Pipx", "https://pipx.pypa.io/stable/", "pypa/pipx"),
    ("Hatch", "https://hatch.pypa.io/latest/", "pypa/hatch"),
    ("PDM", "https://pdm.fming.dev/latest/", "pdm-project/pdm"),
    ("uv", "https://docs.astral.sh/uv/", "astral-sh/uv"),
]

for framework, docs, repo in data:
    stars_count = get_github_stars(repo)
    stars = f"{stars_count / 1000:.1f}k" if stars_count is not None else "⭐ N/A"
    table += f"| {framework} | [🔗 Link]({docs}) | [🔗 Link](https://github.com/{repo}) | {stars} |\n"

print(table)
```

This is considered a problem in the Python community because it means that there is no standard way of managing
dependencies like in other languages like `npm` for `node.js` or `cargo` for `rust` (however, it does seem like `uv` is
trying to become the standard, quickly gaining popularity and being adopted by many projects).

<figure markdown>
![Image](../figures/standards.png){ width="700" }
<figcaption> <a href="https://xkcd.com/927/"> Image credit </a> </figcaption>
</figure>

In the course, we do not care about which package manager you use, but we do care that you use one. If you are already
familiar with one package manager, then skip this exercise and continue to use that. The best recommendation that I can
give regarding package managers, in general, is to find one you like and then stick with it. A lot of time can be wasted
on trying to find the perfect package manager, but in the end, they all do the same with some minor differences.
Check out [this blog post](https://alpopkes.com/posts/python/packaging_tools/) if you want a fairly up-to-date
evaluation of the different environment management and packaging tools that exist in the Python ecosystem.

If you are not familiar with any package managers, then we recommend that you use `conda` and `pip` for this course. You
probably already have [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) installed
on your laptop, which is great. What conda does especially well is that it allows you to create virtual environments
with different Python versions, which can be really useful if you encounter dependencies that have not been updated in
a long time. In this course specifically, we are going to recommend the following workflow

* Use `conda` to create virtual environments with specific Python versions
* Use `pip` to install packages in that environment

Installing packages with `pip` inside `conda` environments has been considered a bad practice for a long time, but
since `conda>=4.6` it is considered safe to do so. The reason for this is that `conda` now has a built-in compatibility
layer that makes sure that `pip` installed packages are compatible with the other packages installed in the environment.

## Python dependencies

Before we get started with the exercises, let's first talk a bit about Python dependencies. One of the most common ways
to specify dependencies in the Python community is through a `requirements.txt` file, which is a simple text file that
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

In general, all packages (should) follow the [semantic versioning](https://semver.org/) standard, which means that the
version number is split into three parts: `x.y.z` where `x` is the major version, `y` is the minor version and `z` is
the patch version.

The reason that we need to specify the version number is that we want to make sure that we can reproduce our code at a
later point. If we do not specify the version number, then we are at the mercy of the package maintainer to not change
the API of the package. This is especially important when working with machine learning models, as we want to make sure
that we can reproduce the exact same model at a later point.

Finally, we also need to discuss *dependency resolution*, which is the process of figuring out which packages are
compatible. This is a non-trivial problem, and there exists a lot of different algorithms for doing this. If you have
ever thought that `pip` and `conda` were taking a long time to install something, then it is probably because they were
trying to figure out which packages are compatible with each other. For example, if you try to install

```bash
pip install "matplotlib >= 3.8.0" "numpy <= 1.19" --dry-run
```

then it would simply fail because there are no versions of `matplotlib` and `numpy` under the given
constraints that are compatible with each other. In this case, we would need to relax the constraints to something like

```bash
pip install "matplotlib >= 3.8.0" "numpy <= 1.21" --dry-run
```

to make it work.

## ❔ Exercises

!!! note "Conda vs. Mamba"

    If you are using `conda` then you can also use `mamba` which is a drop-in replacement `conda` that is faster.
    This means that any `conda` command can be replaced with `mamba` and it should work. Feel free to use `mamba` if
    you are already familiar with `conda` or after having gone through the exercises below. Install instructions can
    be found [here](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html).

For hints regarding how to use `conda` you can check out the
[cheat sheet](https://github.com/SkafteNicki/dtu_mlops/blob/main/s1_development_environment/exercise_files/conda_cheatsheet.pdf)
in the exercise folder.

1. Download and install `conda`. You are free to either install full `conda` or the much simpler version `miniconda`.
    The core difference between the two packages is that `conda` already comes with a lot of packages that you would
    normally have to install with `miniconda`. The downside is that `conda` is a much larger package which can be a
    huge disadvantage on smaller devices. Make sure that your installation is working by writing `conda help` in a
    terminal, and it should show you the help message for conda. If this does not work you probably need to set some
    system variable to point to the
    [conda installation](https://stackoverflow.com/questions/44597662/conda-command-is-not-recognized-on-windows-10)

2. If you have successfully installed conda, then you should be able to execute the `conda` command in a terminal.

    <figure markdown>
    ![Image](../figures/conda_activate.PNG){ width="700" }
    </figure>

    Conda will always tell you what environment you are currently in, indicated by the `(env_name)` in the prompt. By
    default, it will always start in the `(base)` environment.

3. Try creating a new virtual environment. Make sure that it is called `my_environment` and that it installs version
   3.11 of Python. What command should you execute to do this?

    ??? warning "Use Python 3.8 or higher"

        We highly recommend that you use Python 3.8 or higher for this course. In general, we recommend that you use
        the second latest version of Python that is available (currently Python 3.11 as of writing this). This is
        because the latest version of Python is often not supported by all dependencies. You can always check the status
        of different Python version support [here](https://devguide.python.org/versions/).

    ??? success "Solution"

        ```bash
        conda create --name my_environment python=3.11
        ```

4. Which `conda` command gives you a list of all the environments that you have created?

    ??? success "Solution"

        ```bash
        conda env list
        ```

5. Which `conda` command gives you a list of the packages installed in the current environment?

    ??? success "Solution"

        ```bash
        conda list
        ```

    1. How do you easily export this list to a text file? Do this, and make sure you export it to
        a file called `environment.yaml`, as conda uses another format by default than `pip`.

        ??? success "Solution"

            ```bash
            conda list --explicit > environment.yaml
            ```

    2. Inspect the file to see what is in it.

    3. The `environment.yaml` file you have created is one way to secure *reproducibility* between users because
        anyone should be able to get an exact copy of your environment if they have your `environment.yaml` file.
        Try creating a new environment directly from your `environment.yaml` file and check that the packages being
        installed exactly match what you originally had.

        ??? success "Solution"

            ```bash
            conda env create --name <environment-name> --file environment.yaml
            ```

6. As the introduction states, it is fairly safe to use `pip` inside `conda` today. What is the corresponding `pip`
    command that gives you a list of all `pip` installed packages? And how do you export this to `requirements.txt`
    file?

    ??? success "Solution"

        ```bash
        pip list # List all installed packages
        pip freeze > requirements.txt # Export all installed packages to a requirements.txt file
        ```

7. If you look through the requirements that both `pip` and `conda` produce then you will see that it
    is often filled with a lot more packages than what you are using in your project. What you are interested in are the
    packages that you import in your code: `from package import module`. One way to get around this is to use the
    package `pipreqs`, which will automatically scan your project and create a requirements file specific to that.
    Let's try it out:

    1. Install `pipreqs`:

        ```bash
        pip install pipreqs
        ```

    2. Either try out `pipreqs` on one of your own projects or try it out on some other online project.
        What does the `requirements.txt` file `pipreqs` produces look like compared to the files produced
        by either `pip` or `conda`.

## 🧠 Knowledge check

1. Try executing the command

    ```bash
    pip install "pytest < 4.6" pytest-cov==2.12.1
    ```

    based on the error message you get, what would be a compatible way to install these?

    ??? success "Solution"

        As `pytest-cov==2.12.1` requires a version of `pytest` newer than `4.6`, we can simply change the command to be:

        ```bash
        pip install "pytest >= 4.6" pytest-cov==2.12.1
        ```

        but there of course exists other solutions as well.

This ends the module on setting up virtual environments. While the methods mentioned in the exercises are great ways
to construct requirements files automatically, sometimes it is just easier to manually sit down and create the files as you
in that way ensure that only the most necessary requirements are installed when creating a new environment.
