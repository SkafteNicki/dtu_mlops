# 2. Organising your code

## Exercises 

### Conda enviroment

You probably already have conda installed on your laptop, 
https://conda.io/projects/conda/en/latest/user-guide/getting-started.html

### Docker

First you need to [install Docker](https://docs.docker.com/install/).


### Git 

Github is not git! That is very important to state, while github is the domination location for
hosting reposatories. 

Git is a version control system.

It is a requirement for passing this course that you create a public reposatory with your code and
use git to upload any code changes.





### Code organisation

While the two first exercises is about setting up a good enviroment for developing
code, the final exercise here is about organising actual code using reasonable standardized
project structure.

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

	4.2. put your model file into `src/models` folder together with the training file from yesterday.
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
 
### Good cooding practise

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
 
### Extra

While we in this course focus on git/version control for keeping track of code changes, it can also
be extended to keep track of changes to data, model ect. 

In this extra exercise you are encouraged to checkout https://github.com/iterative/dvc which works
similar to git but focuses on version control for machine learning.



