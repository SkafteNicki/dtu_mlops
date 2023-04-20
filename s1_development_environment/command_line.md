![Logo](../figures/icons/terminal.png){ align=right width="130"}

# The terminal

---

!!! info "Core Module"

<figure markdown>
  ![Image](../figures/terminal_power.jpg){ width="500" }
  <figcaption> <a href="https://twitter.com/rorypreddy/status/1257336536477171712"> Image credit </a> </figcaption>
</figure>

Contrary to popular belief, the terminal is not a mythical being that has existed since the dawn of time.
Instead, it was created at a time when it was not given that your computer had a graphical interface that
you could interact with. Think of it as a text interface to your computer.

It is a well-known concept to users of linux, however MAC and (especially) Windows users not so much. Having a basic
understanding of how to use a terminal can really help improve your workflow. We have put a cheat sheet in the
`exercise_files` folder belonging to this session, that gives a quick overview of the different commands that can be
executed in the terminal.

The reason that the terminal is an important tool to get to know, is that doing machine learning in the cloud assumes
that you will interact to some degree with the terminal.

Note if you already are a terminal wizard then feel free to skip the exercises below. They are very elementary.

## Exercises

???+ note "Windows users"

    We highly recommend that you install *Windows Subsystem for Linux* (WSL). This will install a full Linux system
    on your Windows machine. Please follow this
    [guide](https://docs.microsoft.com/en-us/windows/wsl/install-win10). Remember to run commands from an elevated
    (as administrator) Windows Command Prompt. You can in general complete all exercises in the course from a normal
    Windows Command Prompt, but some are easier to do if you run from WSL.

    If you decide to run in WSL you need to remember that you now have two different systems, and install a package
    on one system does not mean that it is installed on the other. For example, if you install `pip` in WSL, you
    need to install it again in Windows if you want to use it there.

    If you decide to not run in WSL, please always work in a Windows Command Prompt and not Powershell.

1. Open a terminal. It should look something like below

    <figure markdown>
    ![Image](../figures/terminal.PNG){ width="1000" }
    </figure>

2. To navigate inside a terminal, we rely on the `cd` command and `pwd` command. Make sure you know how to go back and
    forth in your file system. HINT: try [tab-completion](https://en.wikipedia.org/wiki/Command-line_completion) to
    save some time.

3. The `ls` command is important when we want to know the content of a folder. Try to use the command, and also try
    it with the additional option `-l`. What does it show?

4. Make sure to familiar yourself with the `which`, `echo`, `cat`, `wget`, `less` and `top` commands. Also familiarize
    yourself with the `>` operator. You are probably going to use some of them throughout the course or in your future
    career. For Windows users these commands may be named something else, e.g. `where` command on Windows corresponds
    to `which`.

5. It is also significant that you know how to edit a file through the terminal. Most systems should have the
    `nano` editor installed, else try to figure out which one is installed in your system.

    1. Type `nano` in the terminal

    2. Write the following text in the script

        ```python
        if __name__ == "__main__":
            print("Hello world!")
        ```

    3. Save the script and try to execute it

    4. Afterward, try to edit the file through the terminal (change `Hello world` to something else)

6. All terminals come with their own programming language. The most common system is called `bash`. It can come in handy
    being able to write simple programs in bash. For example, one case is that you want to execute multiple python
    programs sequentially, which can be done through a bash script.

    1. Write a bash script (in `nano`) and try executing it:

        ```bash
        #!/bin/bash
        # A sample Bash script, by Ryan
        echo Hello World!
        ```

    2. Change the bash script to call your python program you just wrote.

    3. Try to Google how to write a simple for-loop that executes the python script 10 times in a row.

## Knowledge check

??? question "Knowledge question 1"

    A common argument that nearly all commands have is the `-h` or `--help` argument. What does it do?

    ??? success "Solution"

        It prints the help message for the command, including subcommands and arguments.
        Try it out by executing ``python --help``.

??? question "Knowledge question 2"

    Another commont argument is the `-v` or `--version` argument. What does it do?

    ??? success "Solution"

        It prints the version of the installed program.
        Try it out by executing `` python --version``.
