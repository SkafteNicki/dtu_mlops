![Logo](../figures/icons/click.png){ align=right width="130"}

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
are all serving a bit different purposes and can therefore be combined in the same project. The three ways are:

## Project scripts

You might already be familiar with the concept of executable scripts. An executable script is a Python script that can
be run directly from the terminal without having to call the Python interpreter. This has been possible for a long time
in Python, by the inclusion of a so-called [shebang](https://en.wikipedia.org/wiki/Shebang_(Unix)) line at the top of
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

Instead of using `argparse` we are here going to look at the [yyper](https://typer.tiangolo.com/) package. `typer`
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

2. Create a new Python file called `greetings.py`. Use the typer package to create a command line interface such
    that running the following lines

    ```bash
    python greetings.py
    python greetings.py --count=3
    python greetings.py --help
    ```

    executes and gives the expected output.

    ??? success "Solution"

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

3. Next lets create a CLI that has subcommands. Add the necessary code such that the following lines can be executed

    ```bash
    python greetings.py hello
    python greetings.py howdy
    ```

    ??? success "Solution"

        ```python
        import typer

        app = typer.Typer()

        @app.command()
        def hello(count: int = 1, name: str = "World"):
            for x in range(count):
                typer.echo(f"Hello {name}!")

        @app.command()
        def howdy(count: int = 1, name: str = "World"):
            for x in range(count):
                typer.echo(f"Howdy {name}!")

        if __name__ == "__main__":
            app()
        ```

5. As an final exercise we provide you with a script that is ready to run as it is, but your job will be do turn it
    into a script with multiple subcommands, with multiple arguments for each subcommand.

    1. Start by taking a look at the provided
        [code](https://github.com/SkafteNicki/dtu_mlops/tree/main/s10_extra/exercise_files/knn_iris.py). It is a simple
        script that runs the K-nearest neighbour classification algorithm on the iris dataset and produces a plot of
        the decision boundary.

    2. Create a script that has the following subcommands with input arguments
        * Subcommand `train`: Load data, train model and save. Should take a single argument `-o` that specifics
            the filename the trained model should be saved to.
        * Subcommand `infer`: Load trained model and runs prediction on input data. Should take two arguments: `-i` that
            specifies which trained model to load and `-d` to specify a user defined datapoint to run inference on.
        * Subcommand `plot`: Load trained model and constructs the decision boundary plot from the code. Should take two
            arguments: `-i` that specifies a trained model to load and `-o` the file to write the generated plot to
        * Subcommand `optim`: Load data, runs hyperparameter optimization and prints optimal parameters. Should at least
            take a single argument that in some way adjust the hyperparameter optimization (free to choose how)

        In the end we like the script to be callable in the following ways

        ```bash
        python main.py train -o 'model.ckpt'
        python main.py infer -i 'model.ckpt' -d [[0,1]]
        python main.py plot -i 'model.ckpt' -o 'generated_plot.png'
        python main.py optim
        ```

6. (Optional) Let's try to combine what we have learned until now. Try to make your `typer` cli into a executable
    script using the `pyproject.toml` file.

    ??? success "Solution"

        ```toml
        [project.scripts]
        greetings = "greetings:app"
        ```

        and remember to install the project in editable mode

        ```bash
        pip install -e .
        ```

        and you should now be able to run the following command in the terminal

        ```bash
        greetings
        ```


##

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

This can be a lot to remember, and it can be easy to make mistakes. To help with this, we are going to look at the
[invoke](http://www.pyinvoke.org/) package. `invoke` is a Python package that allows you to define tasks that can be
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

3. Lets now try to add a task to the `tasks.py` file. Add the following code to the file

    ```python
    from invoke import task

    @task
    def hello(c):
        """ A simple hello world task """  # remember to add docstrings to your tasks, it will show up in the --list
        print("Hello World!")
    ```

    and try to run the following command

    ```bash
    invoke hello
    ```

4. Lets try to create a task that simplifies the process of `git add`, `git commit`, `git push`. Create a task such
    that the following command can be run

    ```bash
    invoke git --message "My commit message"
    ```

    ??? success "Solution"
        ```python
        @task
        def git(c, message):
            c.run(f"git add .")
            c.run(f"git commit -m '{message}'")
            c.run(f"git push")
        ```

5. Create also a command that simplifies the process of bootstrapping a `conda` environment and install the relevant
    dependencies of your project

    ```python
    @task
    def conda(c):
        c.run(f"conda env create -f environment.yml")
        c.run(f"conda activate dtu_mlops")
    ```

    and try to run the following command

    ```bash
    invoke conda
    ```

6. Assuming you have completed the exercises on using `dvc` for version control of data try adding a task that
    simplifies the process of running `dvc repro` for your project

    ```python
    @task
    def dvc(c):
        c.run(f"dvc repro")
    ```

    and try to run the following command

    ```bash
    invoke dvc
    ```


## 🧠 Knowledge check



2. Create a new Python  file `greetings.py` and add the following code:

    ```python
    import click

    @click.command()
    @click.option('--count', default=1, help='Number of greetings.')
    @click.option('--name', prompt='Your name', help='The person to greet.')
    def hello(count, name):
        """Simple program that greets NAME for a total of COUNT times."""
        for x in range(count):
            click.echo(f"Hello {name}!")

    if __name__ == '__main__':
        hello()
    ```

    try running the program in the following ways

    ```bash
    python greetings.py
    python greetings.py --count=3
    python greetings.py --help
    ```

3. Make sure you understand what the `click.command()` decorator and `click.option` decorator does. You can find
    the full API docs [here](https://click.palletsprojects.com/en/8.1.x/api/).

4. As stated above, the power of using a tool like click is due to its ability to define subcommands. In `click` this
    is done through the `click.group()` decorator. To the code example from above, add another command:

    ```python
    @click.command()
    @click.option('--count', default=1, help='Number of greetings.')
    @click.option('--name', prompt='Your name', help='The person to greet.')
    def howdy(count, name):
        for x in range(count):
            click.echo(f"Howdy {name}!")
    ```

    and by using the `click.group()` decorator make these commands into subcommands such that you would be able to
    call the script in the following way

    ```bash
    python greetings.py hello
    python greetings.py howdy
    ```

5. As an final exercise we provide you with a script that is ready to run as it is, but your job will be do turn it
    into a script with multiple subcommands, with multiple arguments for each subcommand.

    1. Start by taking a look at the provided
        [code](https://github.com/SkafteNicki/dtu_mlops/tree/main/s10_extra/exercise_files/knn_iris.py). It is a simple
        script that runs the K-nearest neighbour classification algorithm on the iris dataset and produces a plot of
        the decision boundary.

    2. Create a script that has the following subcommands with input arguments
        * Subcommand `train`: Load data, train model and save. Should take a single argument `-o` that specifics
            the filename the trained model should be saved to.
        * Subcommand `infer`: Load trained model and runs prediction on input data. Should take two arguments: `-i` that
            specifies which trained model to load and `-d` to specify a user defined datapoint to run inference on.
        * Subcommand `plot`: Load trained model and constructs the decision boundary plot from the code. Should take two
            arguments: `-i` that specifies a trained model to load and `-o` the file to write the generated plot to
        * Subcommand `optim`: Load data, runs hyperparameter optimization and prints optimal parameters. Should at least
            take a single argument that in some way adjust the hyperparameter optimization (free to choose how)

        In the end we like the script to be callable in the following ways

        ```bash
        python main.py train -o 'model.ckpt'
        python main.py infer -i 'model.ckpt' -d [[0,1]]
        python main.py plot -i 'model.ckpt' -o 'generated_plot.png'
        python main.py optim
        ```
