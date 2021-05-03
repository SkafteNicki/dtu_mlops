# 2. Getting started with MLOps - Organization

This set of exercises focus on getting organised and make sure that you are familiar with good development
practises. While this may not seem that important, it is crusial when working in large groups that the difference
in how different people organise their code is minimized. Additionally, it is important for the reproduceability
of results to be able to accuratly report the exact enviroment that you are using. Try to think of your computer
as a laboratory. If others were to reproduce your experiements, they would need to know the exact chemicals and
machines that you have been using.

[good-bad-coding](../figures/wtf.jpeg)

A lot of the exercises today are very loosely stated and you are expected to seek out information before you
ask for help (Google is your friend ;-) ).

## Editor 

Notebooks can be great sometimes but for truely getting work done you will need a good editor / IDE. However,
each on 

- Spyder
- Virtual studio code
- PyCharm

### Exercise

1. Download and install one of the editors / IDE and make yourself familiar with it e.g. try out the editor
   on the files that you created in the final exercise in the last lecture.

## Conda enviroment

You probably already have [conda](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) installed 
on your laptop, which is great. Conda is an enviroment manager that helps you make sure that the dependencies of
different projects does not cross-contaminate each other. 

### Exercise

1. Download and install conda. Make sure that your installation is working by writing `conda help` in a terminal 
   and it should show you the help message for conda.

2. Create a new conda enviroment for the remaining of the exercises using `conda create -n "my_enviroment"`

3. Which commando gives you a list of the packages installed in the current enviroment (HINT: check the
   `conda_cheatsheet.pdf` file). How do you easely export this list to a text file?

## Git 

Proper collaboration with other people will require that you can work on the same codebase in a organised manner.
This is the reason that **version control** exist. Simply stated, it is a way to keep track of:

* Who made changes to the code
* When did the change happend
* What changes where made

For a full explaination please see this [page](https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F)

Secondly, it is important to note that Github is not git! Github is the dominating player when it comes to
hosting reposatories but that does not mean that they are the only onces (see [bitbucket](https://bitbucket.org/product/) 
for another example).

That said we will be using git+github throughout this course. It is a requirement for passing this course that 
you create a public reposatory with your code and use git to upload any code changes. How much you choose to
integrate this into your own projects depends, but you are atleast expected to be familiar with git+github.

### Exercise

1. Install git on your computer and make sure that your installation is working by writing `git help` in a 
   terminal and it should show you the help message for git.

2. Create a [github](github.com/) account 

3. In your account create an reposatory, where the intention is that you upload the code from the final exercise
   from yesterday
   
   3.1 After creating the reposatory, clone it to your computer
       ```git clone https://github.com/my_user_name/my_reposatory_name.git```
       
   3.2 Move/copy the three files from yesterday into the reposatory
   
   3.3 Add the files to a commit by using ```git add`` command
   
   3.4 Commit the files using `git commit`
   
   3.5 Finally push the files to your reposatory using `git push`. Make sure to check online that the files
       have been updated in your reposatory.

4. (Optional) Git may seems like a waste of time when solutions like dropbox, google drive ect exist, and it is
   not completly untrue when you are only one or two working on a project. However, these file manegement 
   systems falls short when we hundred to tousand of people work to together. For this exercise you will
   go through the steps of sending an open-source contribution:
   
   4.1 Go online and find a project you do not own, where you can improve the code. For simplicity you can
       just fork this reposatory.
       
   4.2 Clone your local fork of the project using ```git clone```
   
   4.3 As default your local reposatory will be on the ```master branch``` (HINT: you can check this with the
       ```git status``` commando). It is good practise to make a new branch when working on some changes. Use
       the ```git checkout``` commando to create a new branch.
       
   4.4 You are now ready to make changes to reposatory. Try to find something to improve (any spelling mistakes?).
       When you have made the changes, do the standard git cycle: ```add -> commit -> push```
       
   4.5 Go online to the original reposatory and go the ```Pull requests``` tap. Find ```compare``` botton and
       choose the to compare the ```master branch``` of the original repo with the branch that you just created
       in your own repo. Check the diff on the page to make sure that it contains the changes you have made.
       
   4.6 Write a bit about the changes you have made and click send :)


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

In addtion 

4. Install isort
   `pip install isort`
    
5. run isort on your project
   `isort .`
 
## Extra

While we in this course focus on git/version control for keeping track of code changes, it can also
be extended to keep track of changes to data, model ect. 

In this extra exercise you are encouraged to checkout https://github.com/iterative/dvc which works
similar to git but focuses on version control for machine learning.



