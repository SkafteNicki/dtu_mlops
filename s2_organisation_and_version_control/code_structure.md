![Logo](../figures/icons/cookiecutter.png){ align=right width="130"}

# Code organization

---

!!! info "Core Module"

With a basic understanding of version control, it is now time to really begin filling up our code repository. However,
the question then remains how to organize our code? As developers we tend to not think about code organization that
much. It is instead something that just dynamically is being created as we may need it. However, maybe we should spend
some time initially getting organized with the chance of this making our code easier to develop and maintain in the
long run. If we do not spend time organizing our code, we may end up with a mess of code that is hard to understand
or maintain

!!! quote "Big ball of Mud"
    *A Big Ball of Mud is a haphazardly structured, sprawling, sloppy, duct-tape-and-baling-wire, spaghetti-code*
    *jungle. These systems show unmistakable signs of unregulated growth, and repeated, expedient repair. Information*
    *is shared promiscuously among distant elements of the system, often to the point where nearly all the important*
    *information becomes global or duplicated.* <br>
    *The overall structure of the system may never have been well defined.* <br>
    *If it was, it may have eroded beyond recognition. Programmers with a shred of architectural sensibility shun these*
    *quagmires. Only those who are unconcerned about architecture, and, perhaps, are comfortable with the inertia of*
    *the day-to-day chore of patching the holes in these failing dikes, are content to work on such systems.*
    <br> <br>
    Brian Foote and Joseph Yoder, Big Ball of Mud. Fourth Conference on Patterns Languages of Programs
    (PLoP '97/EuroPLoP '97) Monticello, Illinois, September 1997

We are here going to focus on the organization of data science projects e.g. where some kind of data is involved. The
key to modern machine learning/deep learning is without a doubt the vast amounts of data that we have access to today.
It is therefore not unreasonable that data should influence our choice of code structure.

We are in this course going to use the `cookie-cutter` approach. We are not going to argue that `cookie-cutter` is
better than other approaches to code organization, we are just focusing on that it is **standardized** way of creating
project structures. By standardized we mean, that if two persons are both using `cookie-cutter` the layout of their
code does follow some specific rules, making one able to faster get understand the other persons code. Code organization
is therefore not only to make the code easier for you to maintain but also for others to read and understand.

Below is seen the default code structure of cookie-cutter for data science projects.

<figure markdown>
  ![Image](../figures/cookie_cutter.png){ width="1000" }
  <figcaption> <a href="https://github.com/drivendata/cookiecutter-data-science"> Image credit </a> </figcaption>
</figure>

What is important to keep in mind when using a template such as cookie-cutter, is that it exactly is a template. By
definition a template is *guide* to make something. Therefore, not all parts of an template may be important for your
project at hand. Your job is to pick the parts from the template that is useful for organizing your data science.
project.

## ❔ Exercises

After having installed cookiecutter (exercise 1 and 2), the remaining exercises are intended to be used on taking the
simple CNN MNIST classifier from yesterdays exercise and force it into this structure. You are not required to fill out
every folder and file in the project structure, but try to at least follow the steps in exercises. Whenever you need to
run a file I recommend always doing this from the root directory e.g.

```bash
python src/data/make_dataset.py data/raw data/processed
python src/models/train_model.py <arguments>
ect...
```

in this way paths (for saving and loading files) are always relative to the root.

1. Start by reading [this page](https://drivendata.github.io/cookiecutter-data-science/), as it will give you insight
    to why standardized code organization is important.

2. Install [cookie cutter for data science](https://github.com/drivendata/cookiecutter-data-science)

    ``` bash
    # install using the terminal
    pip install cookiecutter
    ```

3. Take a look at the webpage to see how you start a new project. We recommend using `v2` of cookiecutter.

4. After having created your project we are going to install it as a package in our conda environment. Either run

    ```bash
    # install in a terminal in your conda env
    pip install -e .
    # or
    conda develop .
    ```

    In addition you may need to run

    ```bash
    pip install -r requirements.txt
    ```

    to install additional packages required by `cookie-cutter`.

5. Start by filling out the `src/data/make_dataset.py` file. When this file runs, it should take the raw data files in
    `data/raw` (the files that we have provided) process them into a single tensor, normalize the tensor and save this
    intermediate representation to the `data/processed` folder. By normalization here we refer to making sure the
    images have mean 0 and standard deviation 1.

6. Every `cookie-cutter` project comes with a build in `Makefile` that can be used to easily define common operations in
    a project. You do not have to understand the complete file by try taking a look at it. In particular the following
    commands may come in handy

    ```bash
    make data  # runs the make_dataset.py file, try it!
    make clean  # clean __pycache__ files
    make requirements  # install everything in the requirements.py file
    ```

    ??? note "Windows users"

        `make` is a GNU build tool that is by default not available on Windows. There are two recommended ways to get
        it running on Windows. The first is leveraging
        [linux subsystem](https://docs.microsoft.com/en-us/windows/wsl/install-win10) for Windows which you maybe have
        already installed. The second option is utilizing the [chocolatey](https://chocolatey.org/) package manager,
        which enables Windows users to install packages similar to Linux system. The second option is running

7. Put your model file (`model.py`) into `src/models` folder together and insert the relevant code from the `main.py`
    file into the `train_model.py` file. Make sure that whenever a model is trained and it is saved, that it gets saved
    to the `models` folder (preferably in sub-folders).

8. When you run `train_model.py`, make sure that some statistics/visualizations from the trained models gets saved to
    the `reports/figures/` folder. This could be a simple `.png` of the training curve.

9. (Optional) Can you figure out a way to add a `train` command to the `Makefile` such that training can be started
    using

    ```bash
    make train
    ```

10. Fill out the newly created `src/models/predict_model.py` file, such that it takes a pre-trained model file and
    creates prediction for some data. Recommended interface is that users can give this file either a folder with raw
    images that gets loaded in or a `numpy` or `pickle` file with already loaded images e.g. something like this

    ```bash
    python src/models/predict_model.py \
        models/my_trained_model.pt \  # file containing a pretrained model
        data/example_images.npy  # file containing just 10 images for prediction
    ```

11. Fill out the file `src/visualization/visualize.py` with this (as minimum, feel free to add more visualizations)
    * Loads a pre-trained network
    * Extracts some intermediate representation of the data (your training set) from your cnn. This could be the
        features just before the final classification layer
    * Visualize features in a 2D space using
        [t-SNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) to do the dimensionality
        reduction.
    * Save the visualization to a file in the `reports/figures/` folder.

12. (Optional) Feel free to create more files/visualizations (what about investigating/explore the data distribution?)

13. Make sure to update the `README.md` file with a short description on how your scripts should be run

14. Finally make sure to update the `requirements.txt` file with any packages that are necessary for running your
    code (see [this set of exercises](../s1_development_environment/conda.md) for help)

That ends the module on code structure and `cookiecutter`. We again want to stress the point that `cookiecutter` is
just one template for organizing your code. What often happens in a team is that multiple templates are needed in
different stages of the development phase or for different product types because they share commen structure, while
still having some specifics. Keeping templates up-to-date then becomes critical such that no team member is using an
outdated template. If you ever end up in this situation, we highly recommend to checkout
[cruft](https://github.com/cruft/cruft) that works alongside `cookiecutter` to not only make projects but update
existing ones as template evolves. Cruft additionally also has template validation capabilities to ensure projects
match the latest version of a template.
