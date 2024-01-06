![Logo](../figures/icons/git.png){ align=right width="130"}

# Git

---

!!! info "Core Module"

Proper collaboration with other people will require that you can work on the same codebase in an organized manner.
This is the reason that **version control** exist. Simply stated, it is a way to keep track of:

* Who made changes to the code
* When did the change happen
* What changes were made

For a full explanation please see this [page](https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F)

Secondly, it is important to note that GitHub is not git! GitHub is the dominating player when it comes to
hosting repositories but that does not mean that they are the only one providing free repository hosting
(see [bitbucket](https://bitbucket.org/product/) or [gitlab](https://about.gitlab.com/)) for some other examples).

That said we will be using git and GitHub throughout this course. It is a requirement for passing this course that
you create a public repository with your code and use git to upload any code changes. How much you choose to integrate
this into your own projects depends, but you are at least expected to be familiar with git+GitHub.

<figure markdown>
![Image](../figures/git.png){ width="400" }
<figcaption> <a href="https://xkcd.com/1597/"> Image credit </a> </figcaption>
</figure>

## Initial config

!!! quote "What does Git stand for?"

    The name "git" was given by Linus Torvalds when he wrote the very first version. He described the tool as
    "the stupid content tracker" and the name as (depending on your mood):

    * Random three-letter combination that is pronounceable, and not actually used by any common UNIX command. The fact
        that it is a mispronunciation of "get" may or may not be relevant.
    * Stupid. Contemptible and Despicable. simple. Take your pick from the dictionary of slang.
    * "Global information tracker": you're in a good mood, and it actually works for you.
        Angels sing, and a light suddenly fills the room.
    * "Goddamn idiotic truckload of sh*t": when it breaks

1. [Install git](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) on your computer and make sure
    that your installation is working by writing `git help` in a terminal and it should show you the help message for
    git.

2. Create a [GitHub](https://github.com/) account if you do not already have one.

3. To make sure that we do not have to type in our GitHub username every time that we want to do some changes,
    we can once and for all set them on our local machine

    ```bash
    # type in a terminal
    git config credential.helper store
    git config --global user.email <email>
    ```

## Git overview

The most simple way to think of version control, is that it is just nodes with lines connecting them

<figure markdown>
  ![Image](../figures/git_branch.png){width="1000," }
</figure>

Each node, which we call a *commit* is uniquely identified by a hash string. Each node, stores what our code
looked like at that point in time (when we made the commit) and using the hash codes we can easily
revert to a specific point in time.

The commits are made up of local changes that we make to our code. A basic workflow for
adding commits are seen below

<figure markdown>
![Image](../figures/git_structure.PNG){width="1000," }
</figure>

Assuming that we have made some changes to our local *working directory* and that we
want to get these updates to be online in the *remote repository* we have to do the following steps:

* First we run the command `git add`. This will move our changes to the *staging area*. While changes are in the
    staging area we can very easily revert them (using `git restore`). There have therefore not been assigned a unique
    hash to the code yet, and we can therefore still overwrite it.

* To take our code from the *staging area* and make it into a commit, we simply run `git commit` which will locally
    add a note to the graph. It is important again, that we have not pushed the commit to the online *repository* yet.

* Finally, we want others to be able to use the changes that we made. We do a simple `git push` and our
    commit gets online

Of course, the real power of version control is the ability to make branches, as in the image below

<figure markdown>
![Image](../figures/git_branches.png){ width="1000" }
<figcaption>
<a href="https://dev.to/juanbelieni/creating-an-alias-for-deleting-useless-git-branches-105j"> Image credit </a>
</figcaption>
</figure>

Each branch can contain code that are not present on other branches. This is useful when you are many developers
working together on the same project.

## ❔ Exercises

1. In your GitHub account create an repository, where the intention is that you upload the code from the final
    exercise from yesterday

    1. After creating the repository, clone it to your computer

        ```bash
        git clone https://github.com/my_user_name/my_repository_name.git
        ```

    2. Move/copy the three files from yesterday into the repository (and any other that you made)

    3. Add the files to a commit by using `git add` command (1)
       { .annotate }

        1. :man_raising_hand: Writing good commit message is a skill in itself. A commit message should be short but
            informative about the work you are trying to commit. Try to practise writing good commit messages
            throughout the course. You can see
            [this guideline](https://github.com/joelparkerhenderson/git-commit-message) for help.

    4. Commit the files using `git commit`

    5. Finally push the files to your repository using `git push`. Make sure to check online that the files have been
        updated in your repository.

    6. You can always use the command `git status` to check where you are in the process of making a commit.

    7. Also checkout the `git log` command, which will show you the history of commits that you have made.

2. Make sure that you understand how to make branches, as this will allow you to try out code changes without
    messing with your working code. Creating a new branch can be done using:

    ```bash
    # create a new branch
    git checkout -b <my_branch_name>
    ```

    Afterwards, you can use `git checkout` to change between branches (remember to commit your work!)
    Try adding something (a file, a new line of code etc.) to the newly created branch, commit it and
    try changing back to master afterwards. You should hopefully see whatever you added on the branch
    is not present on the main branch.

3. If you do not already have a cloned version of this repository belonging to the course, make sure to make one!
    I am continuously updating/changing some of the material during the course and I therefore recommend that you
    each day before the lecture do a `git pull` on your local copy

4. Git may seem like a waste of time when solutions like dropbox, google drive etc exist, and it is
    not completely untrue when you are only one or two working on a project. However, these file management
    systems falls short when hundreds to thousands of people work together. For this exercise you will
    go through the steps of sending an open-source contribution:

    1. Go online and find a project you do not own, where you can improve the code. You can either look at this
        [page](https://goodfirstissue.dev/) of good issues to get started with or for simplicity you can just choose
        the repository belonging to the course. Now fork the project by clicking the *Fork* button.

        ![forking](../figures/forking.PNG)

        This will create a local copy of the repository which you have complete writing access to. Note that
        code updates to the original repository does not update code in your local repository.

    2. Clone your local fork of the project using `git clone`.

    3. As default your local repository will be on the `main branch` (HINT: you can check this with the
        `git status` command). It is good practice to make a new branch when working on some changes. Use
        the `git branch` command followed by the `git checkout` command to create a new branch.

    4. You are now ready to make changes to the repository. Try to find something to improve (any spelling mistakes?).
        When you have made the changes, do the standard git cycle: `add -> commit -> push`

    5. Go online to the original repository and go to the `Pull requests` tab. Find `compare` button and
        choose the button to compare the `master branch` of the original repo with the branch that you just created
        in your own repository. Check the diff on the page to make sure that it contains the changes you have made.

    6. Write a bit about the changes you have made and click `Create pull request` :)

5. Forking a repository has the consequence that your fork and the repository that you forked can diverge. To
    mitigate this we can set what is called an *remote upstream*. Take a look on this
    [page](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/configuring-a-remote-repository-for-a-fork)
    , and set a remote upstream for the repository you just forked.

6. After setting the upstream branch, we need to pull and merge any update. Take a look on this
    [page](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/syncing-a-fork)
    and figure out how to do this.

7. As a final exercise we want to simulate a *merge conflict*, which happens when two users try to commit changes
    to exactly same lines of code in the codebase, and git is not able to resolve how the different commits should be
    integrated.

    1. In your browser, open your favorite repository (it could be the one you just worked on), go to any file of
        your choosing and click the edit button (see image below) and make some change to the file. For example, if
        you choose a python file you can just import some random packages at the top of the file. Commit the change.

        <figure markdown>
        ![Image](../figures/git_edit.PNG){ width="1000" }
        </figure>

    2. Make sure not to pull the change you just made to your local computer. Locally make changes to the same
        file in the same lines and commit them afterwards.

    3. Now try to `git pull` the online changes. What should (hopefully) happen is that git will tell you that it found
        a merge conflict that needs to be resolved. Open the file and you should see something like this

        ```txt
        <<<<<<< HEAD
        this is some content to mess with
        content to append
        =======
        totally different content to merge later
        >>>>>>> master
        ```

        this should be interpret as: everything that's between `<<<<<<<` and `=======` are the changes made by your
        local commit and everything between `=======` and `>>>>>>>` are the changes you are trying to pull. To fix
        the merge conflict you simply have to make the code in the two "cells" work together. When you are done,
        remove the identifiers `<<<<<<<`, `=======` and `>>>>>>>`.

    4. Finally, commit the merge and try to push.

8. (Optional) The above exercises have focused on how to use git from the terminal, which I highly recommend learning.
    However, if you are using a proper editor they also have build in support for version control. We recommend getting
    familiar with these features (here is a tutorial for
    [VS Code](https://code.visualstudio.com/docs/editor/versioncontrol))

## 🧠 Knowledge check

1. How do you know if a certain directory is a git repository?

    ??? success "Solution"

        You can check if there is a ".git" directory. Alternative you can use the `git status` command.

2. Explain what the file `gitignore` is used for?

    ??? success "Solution"

        The file `gitignore` is used to tell git which files to ignore when doing a `git add .` command. This is
        useful for files that are not part of the codebase, but are needed for the code to run (e.g. data files)
        or files that contain sensitive information (e.g. `.env` files that contain API keys and passwords).

3. You have two branches - *main* and *devel*. What sequence of commands would you need to execute to make sure that
    *devel* is in sync with *main*?

    ??? success "Solution"

        ```bash
        git checkout main
        git pull
        git checkout devel
        git merge main
        ```

4. What best practices are you familiar with regarding version control?

    ??? success "Solution"

        * Use a descriptive commit message
        * Make each commit a logical unit
        * Incorporate others' changes frequently
        * Share your changes frequently
        * Coordinate with your co-workers
        * Don't commit generated files

That covers the basics of git to get you started. In the exercise folder you can find a
[git cheat sheet](https://github.com/SkafteNicki/dtu_mlops/tree/main/s2_organisation_and_version_control/exercise_files/git_cheat_sheet.pdf)
with the most useful commands for future reference. Finally, we want to point out another awesome feature of GitHub:
in browser editor. Sometimes you have a small edit that you want to make, but still would like to do this in a
IDE/editor. Or you may be in the situation where you are working from another device than your usual developer machine.
GitHub has an built-in editor that can simply be enabled by changing any URL from

```bash
https://github.com/username/repository
```

to

```bash
https://github.dev/username/repository
```

Try it out on your newly created repository.
