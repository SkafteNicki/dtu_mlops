---
layout: default
title: M30 - Command Line Interfaces
parent: S10 - Extra
nav_order: 1
---

<img style="float: right;" src="../figures/icons/click.png" width="130">

# Command line interfaces
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

If you have worked with python for some time you are probably familiar with the `argparse` package, which allows you
to directly pass in additional arguments to your script in the terminal

```bash
python my_script.py --arg1 val1 --arg2 val2
```

`argparse` is a very simple way of constructing what is called a command line interfaces (CLI).
[CLI](https://en.wikipedia.org/wiki/Command-line_interface) allows you to interact with your application directly in
the terminal instead of having change things in your code. It is essentially a text-based user interface (UI) (in
contrast to an graphical user interface (GUI) that we know from all our desktop applications).

However, one limitation of `argparse` is the possibility of easily defining an CLI with subcommands. If we take `git`
as an example, `git` is the main command but it has multiple subcommands: `push`, `pull`, `commit` etc. that all can
take their own arguments. This kind of second CLI with subcommands is somewhat possible to do using only `argparse`,
however it requires a bit of hacks.

You could of cause ask the question why we at all would like to have the possibility of defining such CLI. The main
argument here is to give users of our code a single entrypoint to interact with our application instead of having
multiple scripts. As long as all subcommands are proper documented, then our interface should be simple to interact
with (again think `git` where each subcommand can be given the `-h` arg to get specific help).

Instead of using `argparse` we are here going to look at the [click](https://click.palletsprojects.com/en/8.1.x/)
package. `click` extends the functionalities of `argparse` to allow for easy definition of subcommands and many other
things, which we are not going to touch upon in this module. For completeness we should also mention that `click` is not
the only package for doing this, and of other excellent frameworks for creating command line interfaces easily we can
mention [Typer](https://typer.tiangolo.com/).

## Exercises

{: .highlight }
> [Exercise files](https://github.com/SkafteNicki/dtu_mlops/tree/main/s10_extra/exercise_files)

1. Install [click](https://click.palletsprojects.com/en/8.1.x/)

   ```bash
   pip install click
   ```

2. Create a new python file `greetings.py` and add the following code:

   ```python
   import click

   @click.command()
   @click.option('--count', default=1, help='Number of greetings.')
   @click.option('--name', prompt='Your name',
                 help='The person to greet.')
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
   @click.option('--name', prompt='Your name',
                 help='The person to greet.')
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

   1. Start by taking a look at the provided [code](exercise_files/knn_iris.py). It is a simple script that runs the
      K-nearest neighbour classification algorithm on the iris dataset and produces a plot of the decision boundary.

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
