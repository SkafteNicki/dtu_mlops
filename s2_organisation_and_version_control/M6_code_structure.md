---
layout: default
title: M6 - Code structure
parent: S2 - Organisation and version control
nav_order: 2
---

# Code organization
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

With a basic understanding of version control, it is now time to really begin filling up our code repository. However, the question then remains how to organize our code? As developers we tend to not think about code organization that much. It is instead something that just dynamically is being created as we may need it. However, maybe we should spend some time initially getting organized with the chance of this making our code easier to develop and maintain in the long run.

We are here going to focus on the organization of data science projects e.g. where some kind of data is involved. The key to modern machine learning/deep learning is without a doubt the vast amounts of data that we have access to today. It is therefore not unreasonable that data should influence our choice of code structure.

We are in this course going to use the `cookie-cutter` approach. We are not going to argue that `cookie-cutter` is better than other approaches to code organization, we are just focusing on that it is **standardized** way of creating project structures. By standardized we mean, that if two persons are both using `cookie-cutter` the layout of their code does follow some specific rules, making one able to faster get understand the other persons code. Code organization is therefore not only to make the code easier for you to maintain but also for others to read and understand.

### Exercise

1. Start by reading [this page](https://drivendata.github.io/cookiecutter-data-science/), as it will give you insight to why standardized code organization is important.

2. Install [cookie cutter for data science](https://github.com/drivendata/cookiecutter-data-science)
   ``` bash
   # install using the terminal
   pip install cookiecutter
   ```

3. Take a look at the webpage to see how you start a new project.

  The remaining of this exercise is intended to be used on taking the simple cnn mnist classifier from yesterdays exercise and force it into this structure. You are not required to fill out every folder and file in the project structure, but complete the following steps. When you need to run a file I recommend always doing this from the root directory
  e.g.
  ```bash
  python src/data/make_dataset.py data/raw data/processed
  python src/models/train_model.py <arguments>
  ect...
  ```
  in this way paths (for saving and loading files) are always relative to the root.

4. After having created your project we are going to install it as a package in our conda enviroment. Either run 
    ```bash
    # install in a terminal in your conda env
    pip install -e .
    # or 
    conda develop .
	```

5. Start by filling out the `src/data/make_dataset.py` file. When this file runs, it should take the raw data files in `data/raw` (the files that we have provided) process them into a single tensor, normalize the tensor and save this intermediate representation to the `data/processed` folder. 

5. Every `cookie-cutter` project comes with a build in `Makefile` that can be used to easily define common operations in a project. You do not have to understand the complete file by try taking a look at it. In particular the following commands may come in handy
    ```bash
	make data  # runs the make_dataset.py file, try it!
	make clean  # clean __pycache__ files
	make requirements  # install everything in the requirements.py file
	```
    If you are running Windows, `make` is not a build-in command and you either need to install [chocolatey](https://chocolatey.org/) or [linux subsystem](https://docs.microsoft.com/en-us/windows/wsl/install-win10) for Windows.

6. Put your model file (`model.py`) into `src/models` folder together and insert the relevant code from the `main.py` file into the `train_model.py` file. Make sure that whenever a model is trained and it is saved, that it gets saved to the `models` folder (preferably in sub-folders).

7. When you run `train_model.py`, make sure that some statistics/visualizations from the trained models gets saved to the `reports/figures/` folder. This could be a simple `.png` of the training curve. 

8. (Optional) Can you figure out a way to add a `train` command to the `Makefile` such that training can be started using
    ```bash
	make train
	```

9. Fill out the newly created `src/models/predict_model.py` file, such that it takes a pre-trained model file and creates prediction for some data. Recommended interface is that users can give this file either a folder with raw images that gets loaded in or a `numpy` or `pickle` file with already loaded images e.g. something like this
    ```bash
	python src/models/predict_model.py \
	    models/my_trained_model.pt \  # file containing a pretrained model
		data/example_images.npy  # file containing just 10 images for prediction
    ```
9. Fill out the file `src/visualization/visualize.py` with this (as minimum, feel free to add more vizualizations)
	- loads a pre-trained network,
	- extracts some intermediate representation of the data (your training set) from your cnn. This could be the features just before the final classification layer
	- Visualize features in a 2D space using [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) to do the dimensionality reduction
	- save the visualization to a file in the `reports/figures/` folder

10. (Optional) Feel free to create more files/visualizations (what about investigating/explore the data distribution?)

11. Make sure to update the `README.md` file with a short description on how your scripts should be run

12. Finally make sure to update the `requirements.txt` file with any packages that are necessary for running your code (see [this set of exercises](../s1_getting_started/M2_conda.md) for help)
