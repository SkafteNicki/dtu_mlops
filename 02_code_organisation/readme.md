# 2. Getting started with MLOps - Organization

This set of exercises focus on getting organized and make sure that you are familiar with good development
practices. While this may not seem that important, it is crucial when working in large groups that the difference
in how different people organize their code is minimized. Additionally, it is important for the reproducibility
of results to be able to accurately report the exact environment that you are using. Try to think of your computer
as a laboratory. If others were to reproduce your experiments, they would need to know the exact configuration of your
machine that you have been using, similar to how a real laboratory needs to report the exact chemicals they are using.
This is one of the cornerstones of the [scientific method](https://en.wikipedia.org/wiki/Scientific_method)

<p align="center">
  <img src="../figures/wtf.jpeg" width="700" title="hover text">
</p>
(All credit to [link](https://the-tech-lead.com/2020/07/19/good-code-bad-code/))

A lot of the exercises in this course are very loosely stated (including the exercises today). You are expected to
seek out information before you ask for help (Google is your friend!) as you will both learn more for trying to
solve the problems yourself and it is more realistic of how the "real world" works.

## Editor 

Notebooks can be great for testing out ideas, developing simple code and explaining and visualizing certain aspects
of a codebase. Remember that [Jupyter notebook](https://jupyter.org/) was created with intention to "...allows you 
to create and share documents that contain live code, equations, visualizations and narrative text." However, 
any larger deep learning project will require you to work in multiple `.py` files and here notebooks will provide 
a suboptimal workflow. Therefore, to for truly getting "work done" you will need a good editor / IDE. 

Many opinions exist on this matter, but for simplicity we recommend getting started with one of the following 3:

Editor		   | Webpage  				| Comment (Biased opinion)
-------------------|------------------------------------|----------------------------------------------------------------------
Spyder             | https://www.spyder-ide.org/        | Matlab like enviroment that is easy to get started with
Visual studio code | https://code.visualstudio.com/     | Support for multiple languages with fairly easy setup
PyCharm            | https://www.jetbrains.com/pycharm/ | IDE for python professionals. Will take a bit of time getting used to
--------------------------------------------------------------------------------------------------------------------------------

### Exercise

1. Download and install one of the editors / IDE and make yourself familiar with it e.g. try out the editor
   on the files that you created in the final exercise in the last lecture.

## Conda environment

You probably already have [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) installed 
on your laptop, which is great. Conda is an environment manager that helps you make sure that the dependencies of
different projects does not cross-contaminate each other. 

### Exercise

1. Download and install conda. Make sure that your installation is working by writing `conda help` in a terminal 
   and it should show you the help message for conda.

2. Create a new conda environment for the remaining of the exercises using `conda create -n "my_enviroment"`

3. Which commando gives you a list of the packages installed in the current environment (HINT: check the
   `conda_cheatsheet.pdf` file). How do you easily export this list to a text file?

## Git 

Proper collaboration with other people will require that you can work on the same codebase in a organized manner.
This is the reason that **version control** exist. Simply stated, it is a way to keep track of:

* Who made changes to the code
* When did the change happen
* What changes where made

For a full explanation please see this [page](https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F)

Secondly, it is important to note that Github is not git! Github is the dominating player when it comes to
hosting repositories but that does not mean that they are the only once (see [bitbucket](https://bitbucket.org/product/) 
for another example).

That said we will be using git+github throughout this course. It is a requirement for passing this course that 
you create a public repository with your code and use git to upload any code changes. How much you choose to
integrate this into your own projects depends, but you are at least expected to be familiar with git+github.

### Exercise

1. Install git on your computer and make sure that your installation is working by writing `git help` in a 
   terminal and it should show you the help message for git.

2. Create a [github](github.com/) account 

3. In your account create an repository, where the intention is that you upload the code from the final exercise
   from yesterday
   
   3.1 After creating the repository, clone it to your computer
       ```git clone https://github.com/my_user_name/my_reposatory_name.git```
       
   3.2 Move/copy the three files from yesterday into the repository
   
   3.3 Add the files to a commit by using ```git add`` command
   
   3.4 Commit the files using `git commit`
   
   3.5 Finally push the files to your reposatory using `git push`. Make sure to check online that the files
       have been updated in your reposatory.

4. If you do not already have a cloned version of this reposatory, make sure to make one! I am continuously updating/
   changing some of the material and I therefore recommend that you each day before the lecture do a `git pull` on your
   local copy

5. Git may seems like a waste of time when solutions like dropbox, google drive ect exist, and it is
   not completly untrue when you are only one or two working on a project. However, these file manegement 
   systems falls short when we hundred to thousand of people work to together. For this exercise you will
   go through the steps of sending an open-source contribution:
   
   5.1 Go online and find a project you do not own, where you can improve the code. For simplicity you can
       just choose the reposatory belonging to the course. Now fork the project by clicking the *Fork* botton.
       ![forking](../figures/forking.PNG)
       This will create a local copy of the reposatory which you have complete writing access to. Note that
       code updates to the original reposatory does not update code in your local reposatory.

   5.2 Clone your local fork of the project using ```git clone```

   5.3 As default your local reposatory will be on the ```master branch``` (HINT: you can check this with the
       ```git status``` commando). It is good practise to make a new branch when working on some changes. Use
       the ```git checkout``` commando to create a new branch.

   5.4 You are now ready to make changes to reposatory. Try to find something to improve (any spelling mistakes?).
       When you have made the changes, do the standard git cycle: ```add -> commit -> push```

   5.5 Go online to the original reposatory and go the ```Pull requests``` tap. Find ```compare``` botton and
       choose the to compare the ```master branch``` of the original repo with the branch that you just created
       in your own repo. Check the diff on the page to make sure that it contains the changes you have made.

   5.6 Write a bit about the changes you have made and click send :)

## Code organisation

While the two first exercises is about setting up a good enviroment for developing
code, the final exercise here is about organising actual code using reasonable standardized
project structure.

### Exercise

1. Start by reading, as it will give you insight to why standardized code organisation is important:
https://drivendata.github.io/cookiecutter-data-science/

2. Install [cookie cutter for data science](https://github.com/drivendata/cookiecutter-data-science)

```
pip install cookiecutter
```

3. Take a look at the webpage to see how you start a new project.

4. The remaining of this exercise is intended to be used on taking the simple cnn mnist classifier from
yesterdays exercise and force it into this structure. You are not required to fill out every folder in
the project structure, but complete the following steps

	4.1. start by filling out the `src/data/make_dataset.py` file. Make sure that data gets correctly
	filled into the `data/raw` and `data/processed` folder. Hint: `torchvision.datasets.MNIST` will
	make everything smooth for you.

	4.2. put your model file into `src/models` folder together with the training file (`main.py`) from yesterday.
	Make sure that trained models gets saved in the `models` folder (preferrable in subfolders).
	
	4.3 make sure that some statistics from the trained models gets saved to the `reports/figures/`
	folder. This could be a simple .png of the training curve. 

	4.4 create a new file `scr/models/predict_model.py` that takes a pre-trained model file and
	creates prediction for some data. Recommended interface is that users can give this file either 
	a folder with raw images that gets loaded in or a `numpy` or `pickle` file with already loaded
	images

	4.5 create a new file `scr/visualization/visualize.py` that as minimum does the following
	- loads a pretrained network, extracts features from the mnist test set (i.e. the features
	just before the final classification layer and does [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
	embedding of the features (color coded according to the class label).
	- feel free to create more files/more visualizations (what about investigating/explore the data
	distribution of mnist?)

	4.6 make sure to update the readme with a short description on how your scripts should be run

	4.7 finally make sure to update the `requirements.txt` file with any packages that are nessesary
	for running your code.
 
## Good cooding practise

While python already enforces some styling (e.g. code should be indented in a specific way), this is not enough
to secure that code from different users actually look like each other. Code being written in a specific style
is important when doing a project together with multiple people. The question then remains what styling you
should use. This is where [Pep8](https://www.python.org/dev/peps/pep-0008/) comes into play, which is the 
official style guide for python. It is essentially contains what is considered "good practise" and 
"bad practise" when coding python. One way to check if your code is pep8 compliant is to use 
[flake8](https://flake8.pycqa.org/en/latest/).

### Exercises

1. Install flake8
   ```
   pip install flake8
   ```

2. run flake8 on your project
   ```
   flake8 .
   ```
   are you pep8 compliant or are you a normal mortal?

You could go and fix all the small errors that `flake8` is giving. However, in practise large projects instead
relies on some kind of code formatter, that will automatically format your code for you to be pep8 compliant.
Some of the biggest are:

* [black](https://github.com/psf/black)
* [yapf](https://github.com/google/yapf)

3. install a code formatter of your own choice and let it format atleast one of the script in your codebase.
   (Optional): play around with different formatters a find out which formatter you like the most.

One aspect not covered by `pep8` is how `import` statements in python should be organised. If you are like most
people, you place your `import` statements at the top of the file and they are ordered simply by when you needed them.
For this reason `import` statements is something we also want to take care of, but do not want to deal with outself.

4. Install [isort](https://github.com/PyCQA/isort) the standard for sorting imports
   ``` 
   pip install isort
   ```

5. run isort on your project
   ```
   isort .
   ```

## Extra

While we in this course focus on git/version control for keeping track of code changes, it can also
be extended to keep track of changes to data, model ect. 

In this extra exercise you are encouraged to checkout https://github.com/iterative/dvc which works
similar to git but focuses on version control for machine learning.
