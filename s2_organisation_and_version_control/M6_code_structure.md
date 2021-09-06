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

While the two first exercises is about setting up a good environment for developing
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

	4.4 create a new file `src/models/predict_model.py` that takes a pre-trained model file and
	creates prediction for some data. Recommended interface is that users can give this file either 
	a folder with raw images that gets loaded in or a `numpy` or `pickle` file with already loaded
	images

	4.5 create a new file `src/visualization/visualize.py` that as minimum does the following
	- loads a pretrained network, extracts features from the mnist test set (i.e. the features
	just before the final classification layer and does [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)
	embedding of the features (color coded according to the class label).
	- feel free to create more files/more visualizations (what about investigating/explore the data
	distribution of mnist?)

	4.6 make sure to update the readme with a short description on how your scripts should be run

	4.7 finally make sure to update the `requirements.txt` file with any packages that are nessesary
	for running your code.
 


