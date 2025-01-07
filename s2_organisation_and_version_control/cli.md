![Logo](../figures/icons/typer.png){ align=right width="130"}

# Command line interfaces

---

As we already laid out in the very first [module](../s1_development_environment/command_line.md), the command line is a
powerful tool for interacting with your computer. You should already now be familiar with running basic Python commands
in the terminal:

```bash
python my_script.py
```

However, as your projects grow in size and complexity, you will often find yourself in need of more advanced ways of
interacting with your code. This is where [command line interface](https://en.wikipedia.org/wiki/Command-line_interface)
(CLI) comes into play. A CLI can be seen as a way for you to define the user interface of your application directly in
the terminal. Thus, there is no right or wrong way of creating a CLI, it is all about what makes sense for your
application.

In this module we are going to look at three different ways of creating a CLI for your machine learning projects. They
are all serving a bit different purposes and can therefore be combined in the same project. However, you will most
likely also feel that they are overlapping in some areas. That is completely fine, and it is up to you to decide which
one to use in which situation.

## Project scripts

You might already be familiar with the concept of executable scripts. An executable script is a Python script that can
be run directly from the terminal without having to call the Python interpreter. This has been possible for a long time
in Python, by the inclusion of a so-called [shebang](https://en.wikipedia.org/wiki/hash-bang) line at the top of
the script. However, we are going to look at a specific way of defining
[executable scripts](https://packaging.python.org/en/latest/guides/writing-pyproject-toml/#creating-executable-scripts)
using the standard `pyproject.toml` file, which you should have learned about in this
[module](code_structure.md).

### ❔ Exercises

1. We are going to assume that you have a training script in your project that you would like to be able to run from the
    terminal directly without having to call the Python interpreter. Lets assume it is located like this

    ```plaintext
    src/
    ├── my_project/
    │   ├── __init__.py
    │   ├── train.py
    pyproject.toml
    ```

    In your `pyproject.toml` file add the following lines. You will need to alter the paths to match your project.

    ```toml
    [project.scripts]
    train = "my_project.train:main"
    ```

    what do you think the `train = "my_project.train:main"` line do?

    ??? success "Solution"

        The line tells Python that we want to create an executable script called `train` that should run the `main`
        function in the `train.py` file located in the `my_project` package.

2. Now, all that is left to do is install the project again in editable mode

    ```bash
    pip install -e .
    ```

    and you should now be able to run the following command in the terminal

    ```bash
    train
    ```

    Try it out and see if it works.

3. Add additional scripts to your `pyproject.toml` file that allows you to run other scripts in your project from the
    terminal.

    ??? success "Solution"

        We assume that you also have a script called `evaluate.py` in the `my_project` package.

        ```toml
        [project.scripts]
        train = "my_project.train:main"
        evaluate = "my_project.evaluate:main"
        ```

That is all there really is to it. You can now run your scripts directly from the terminal without having to call the
Python interpreter. Some good examples of Python packages that uses this approach are
[numpy](https://github.com/numpy/numpy/blob/main/pyproject.toml#L43-L45),
[pylint](https://github.com/pylint-dev/pylint/blob/main/pyproject.toml#L67-L71) and
[kedro](https://github.com/kedro-org/kedro/blob/main/pyproject.toml#L99-L100).

## Command line arguments

If you have worked with Python for some time you are probably familiar with the `argparse` package, which allows you
to directly pass in additional arguments to your script in the terminal

```bash
python my_script.py --arg1 val1 --arg2 val2
```

`argparse` is a very simple way of constructing what is called a command line interfaces. However, one limitation of
`argparse` is the possibility of easily defining an CLI with subcommands. If we take `git` as an example, `git` is the
main command but it has multiple subcommands: `push`, `pull`, `commit` etc. that all can take their own arguments. This
kind of second CLI with subcommands is somewhat possible to do using only `argparse`, however it requires a bit of
hacks.

You could of course ask the question why we at all would like to have the possibility of defining such CLI. The main
argument here is to give users of our code a single entrypoint to interact with our application instead of having
multiple scripts. As long as all subcommands are proper documented, then our interface should be simple to interact
with (again think `git` where each subcommand can be given the `-h` arg to get specific help).

Instead of using `argparse` we are here going to look at the [typer](https://typer.tiangolo.com/) package. `typer`
extends the functionalities of `argparse` to allow for easy definition of subcommands and many other things, which we
are not going to touch upon in this module. For completeness we should also mention that `typer` is not the only package
for doing this, and of other excellent frameworks for creating command line interfaces easily we can mention
[click](https://click.palletsprojects.com/en/8.1.x/).

### ❔ Exercises

1. Start by installing the `typer` package

    ```bash
    pip install typer
    ```

    remember to add the package to your `requirements.txt` file.

2. To get you started with `typer`, let's just create a simple hello world type of script. Create a new Python file
    called `greetings.py` and use the `typer` package to create a command line interface such that running the
    following lines

    ```bash
    python greetings.py            # should print "Hello World!"
    python greetings.py --count=3  # should print "Hello World!" three times
    python greetings.py --help     # should print the help message, informing the user of the possible arguments
    ```

    executes and gives the expected output. Relevant [documentation](https://typer.tiangolo.com/).

    ??? success "Solution"

        Importantly for `typer` is that you need to provide
        [type hints](https://typer.tiangolo.com/tutorial/#python-types) for the arguments. This is because `typer` needs
        these to be able to work properly.

        ```python
        import typer
        app = typer.Typer()

        @app.command()
        def hello(count: int = 1, name: str = "World"):
            for x in range(count):
                typer.echo(f"Hello {name}!")

        if __name__ == "__main__":
            app()
        ```

3. Next, lets try on a bit harder example. Below is a simple script that trains a support vector machine on the iris
    dataset.

    !!! example "iris_classifier.py"

        ```python linenums="1" title="iris_classifier.py"
        --8<-- "s2_organisation_and_version_control/exercise_files/typer_exercise.py"
        ```

    Implement a CLI for the script such that the following commands can be run

    ```bash
    python iris_classifier.py train --output 'model.ckpt'  # should train the model and save it to 'model.ckpt'
    python iris_classifier.py train -o 'model.ckpt'  # should be the same as above
    ```

    ??? success "Solution"

        We are here making use of the
        [short name option](https://typer.tiangolo.com/tutorial/options/name/#cli-option-short-names) in typer for
        giving an shorter alias to the `--output` option.

        ```python linenums="1" title="iris_classifier.py"
        --8<-- "s2_organisation_and_version_control/exercise_files/typer_exercise_solution.py"
        ```

4. Next lets create a CLI that has more than a single command. Continue working in the basic machine learning
    application from the previous exercise, but this time we want to define two separate commands

    ```bash
    python iris_classifier.py train --output 'model.ckpt'
    python iris_classifier.py evaluate 'model.ckpt'
    ```

    ??? success "Solution"

        The only key difference between the two is that in the `train` command we define the `output` argument to
        to be an optional parameter e.g. we provide a default and for the `evaluate` command it is a required parameter.

        ```python linenums="1" title="iris_classifier.py"
        --8<-- "s2_organisation_and_version_control/exercise_files/typer_exercise_solution2.py"
        ```

5. Finally, let's try to define subcommands for our subcommands e.g. something similar to how `git` has the subcommand
    `remote` which in itself has multiple subcommands like `add`, `rename` etc. Continue on the simple machine
    learning application from the previous exercises, but this time define a cli such that

    ```bash
    python iris_classifier.py train svm --kernel 'linear'
    python iris_classifier.py train knn -k 5
    ```

    e.g the `train` command now has two subcommands for training different machine learning models (in this case SVM
    and KNN) which each takes arguments that are unique to that model. Relevant
    [documentation](https://typer.tiangolo.com/tutorial/subcommands/).

    ??? success

        ```python linenums="1" title="iris_classifier.py"
        --8<-- "s2_organisation_and_version_control/exercise_files/typer_exercise_solution3.py"
        ```

6. (Optional) Let's try to combine what we have learned until now. Try to make your `typer` cli into a executable
    script using the `pyproject.toml` file and try it out!

    ??? success "Solution"

        Assuming that our `iris_classifier.py` script from before is placed in `src/my_project` folder, we should just
        add

        ```toml
        [project.scripts]
        greetings = "src.my_project.iris_classifier:app"
        ```

        and remember to install the project in editable mode

        ```bash
        pip install -e .
        ```

        and you should now be able to run the following command in the terminal

        ```bash
        iris_classifier train knn
        ```

This covers the basic of `typer` but feel free to deep dive into how the package can help you custimize your CLIs.
Checkout [this page](https://typer.tiangolo.com/tutorial/printing/) on adding colors to your CLI or
[this page](https://typer.tiangolo.com/tutorial/parameter-types/number/) on validating the inputs to your CLI.

## Non-Python code

The two sections above have shown you how to create a simple CLI for your Python scripts. However, when doing machine
learning projects, you often have a lot of non-Python code that you would like to run from the terminal. Based on the
learning modules you have already completed, you have already encountered a couple of CLI tools that are used in our
projects:

* [conda](../s1_development_environment/package_manager.md) for managing environments
* [git](git.md) for version control of code
* [dvc](dvc.md) for version control of data

As we begin to move into the next couple of learning modules, we are going to encounter even more CLI tools that we need
to interact with. Here is a example of long command that you might need to run in your project in the future

```bash
docker run -v $(pwd):/app -w /app --gpus all --rm -it my_image:latest python my_script.py --arg1 val1 --arg2 val2
```

This can be a lot to remember, and it can be easy to make mistakes. Instead it would be nice if we could just do

```bash
run my_command --arg1=val1 --arg2=val2
```

e.g. easier to remember because we have remove a lot of the hard-to-remember stuff, but we are still able to configure
it to our liking. To help with this, we are going to look at the [invoke](http://www.pyinvoke.org/) package.
`invoke` is a Python package that allows you to define tasks that can be
run from the terminal. It is a bit like a more advanced version of the [Makefile](https://makefiletutorial.com/) that
you might have encountered in other programming languages. Some good alternatives to `invoke` are
[just](https://github.com/casey/just) and [task](https://github.com/go-task/task), but we have chosen to focus on
`invoke` in this module because it can be installed as a Python package making installation across different systems
easier.

### ❔ Exercises

1. Start by installing `invoke`

    ```bash
    pip install invoke
    ```

    remember to add the package to your `requirements.txt` file.

2. Add a `tasks.py` file to your repository and try to just run

    ```bash
    invoke --list
    ```

    which should work but inform you that no tasks are added yet.

3. Let's now try to add a task to the `tasks.py` file. The way to do this with invoke is to import the `task`
    decorator from `invoke` and then decorate a function with it:

    ```python
    from invoke import task
    import os

    @task
    def python(ctx):
        """ """
        ctx.run("which python" if os.name != "nt" else "where python")

    ```

    the first argument of any task-decorated function is the `ctx` context argument that implements the `run` method
    for running any command as we run them in the terminal. In this case we have simply implemented a task that
    returns the current Python interpreter but it works for all operating systems. Check that it works by running:

    ```bash
    invoke python
    ```

4. Lets try to create a task that simplifies the process of `git add`, `git commit`, `git push`. Create a task such
    that the following command can be run

    ```bash
    invoke git --message "My commit message"
    ```

    Implement it and use the command to commit the taskfile you just created!

    ??? success "Solution"

        ```python
        @task
        def git(ctx, message):
            ctx.run(f"git add .")
            ctx.run(f"git commit -m '{message}'")
            ctx.run(f"git push")
        ```

5. As you have hopefully realized by now, the most important method in `invoke` is the `ctx.run` method which actually
    run the commands you want to run in the terminal. This command takes multiple additional arguments. Try out the
    arguments `warn`, `pty`, `echo` and explain in your own words what they do.

    ??? success "Solution"

        * `warn`: If set to `True` the command will not raise an exception if the command fails. This can be useful if
            you want to run multiple commands and you do not want the whole process to stop if one of the commands fail.
        * `pty`: If set to `True` the command will be run in a pseudo-terminal. If you want to enable this or not,
            depends on the command you are running.
            [Here](https://www.pyinvoke.org/faq.html#why-is-my-command-behaving-differently-under-invoke-versus-being-run-by-hand)
            is a good explanation of when/why you should use it.
        * `echo`: If set to `True` the command will be printed to the terminal before it is run.

6. Create a command that simplifies the process of bootstrapping a `conda` environment and install the relevant
    dependencies of your project.

    ??? success "Solution"

        ```python
        @task
        def conda(ctx, name: str = "dtu_mlops"):
            ctx.run(f"conda env create -f environment.yml", echo=True)
            ctx.run(f"conda activate {name}", echo=True)
            ctx.run(f"pip install -e .", echo=True)
        ```

        and try to run the following command

        ```bash
        invoke conda
        ```

7. Assuming you have completed the exercises on using [dvc](dvc.md) for version control of data, lets also try to add
    a task that simplifies the process of adding new data. This is the list of commands that need to be run to add new
    data to a dvc repository: `dvc add`, `git add`, `git commit`, `git push`, `dvc push`. Try to implement a task
    that simplifies this process. It needs to take two arguments for defining the folder to add and the commit message.

    ??? success "Solution"

        ```python
        @task
        def dvc(ctx, folder="data", message="Add new data"):
            ctx.run(f"dvc add {folder}")
            ctx.run(f"git add {folder}.dvc .gitignore")
            ctx.run(f"git commit -m '{message}'")
            ctx.run(f"git push")
            ctx.run(f"dvc push")
        ```

        and try to run the following command

        ```bash
        invoke dvc --folder 'data' --message 'Add new data'
        ```

8. As the final exercise, lets try to combine every way of defining CLIs we have learned about in this module. Define
    a task that does the following

    * calls `dvc pull` to download the data
    * calls a entrypoint `my_cli` with the subcommand `train` with the arguments `--output 'model.ckpt'`

    ??? success "Solution"

        ```python
        from invoke import task

        @task
        def pull_data(ctx):
            ctx.run("dvc pull")

        @task(pull_data)
        def train(ctx)
            ctx.run("my_cli train")
        ```

That is all there is to it. You should now be able to define tasks that can be run from the terminal to simplify the
process of running your code. We recommend that as you go through the learning modules in this course that you slowly
start to add tasks to your `tasks.py` file that simplifies the process of running the code you are writing.

## 🧠 Knowledge check

1. What is the purpose of a command line interface?

    ??? success "Solution"

        A command line interface is a way for you to define the user interface of your application directly in the
        terminal. It allows you to interact with your code in a more advanced way than just running Python scripts.
