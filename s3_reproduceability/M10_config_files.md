---
layout: default
title: M10 - Config files
parent: S3 - Reproduceability
nav_order: 2
---

# Config files
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

With docker we can make sure that our compute environment is reproducible, but that does not mean that all our experiments magically becomes reproducible. There are other factors that are important for creating reproducible experiments.

In [this paper](https://arxiv.org/abs/1909.06674) (highly recommended read) the authors tried to reproduce the results of 255 papers and tried to figure out which factors where significant to succeed. One of those factors were "Hyperparameters Specified" e.g. whether or not the authors of the paper had precisely specified the hyperparameter that was used to run the experiments. It should come as no surprise that this can be a determining factor for reproducibility, however it is not given that hyperparameters is always well specified.

### Configure experiments

There is really no way around it: deep learning contains *a lot* of hyperparameters. In general, a *hyperparameter* is any parameter that affects the learning process (e.g. the weights of a neural network are not hyperparameters because they are a consequence of the learning process). The problem with having many hyperparameters to control in your code, is that if you are not careful and structure them it may be hard after running a experiment to figure out which hyperparameters were actually used. Lack of proper configuration management can cause serious problems with reliability, uptime, and the ability to scale a system.

One of the most basic ways of structuring hyperparameters, is just to put them directly into you `train.py` script in some object:

```python
class my_hp:
    batch_size: 64
    lr: 128
    other_hp: 12345

# easy access to them
dl = DataLoader(Dataset, batch_size=my_hp.batch_size)
```

the problem here is configuration is not easy. Each time you want to run a new experiment, you basically have to change the script. If you run the code multiple times, without committing the changes in between then the exact hyperparameter configuration for some experiments may be lost. Alright, with this in mind you change strategy to use an [argument parser](https://docs.python.org/3/library/argparse.html) e.g. run experiments like this

```bash
python train.py --batch_size 256 --learning_rate 1e-4 --other_hp 12345
```

This at least solves the problem with configurability. However, we again can end up with loosing experiments if we are not carefull.

What we really want is some way to easy configure our experiments where the hyperparameters are systematically saved with the experiment. For this we turn our attention to [Hydra](https://hydra.cc/), a configuration tool from Facebook. Hydra operates on top of [OmegaConf](https://github.com/omry/omegaconf) which is a `yaml` based hierarchical configuration system.

A simple `yaml` configuration file could look like
```yaml
#config.yaml
hyperparameters:
  batch_size: 64
  learning_rate: 1e-4
```
with the corresponding python code for loading the file
```python
from omegaconf import OmegaConf
# loading
config = OmegaConf.load('config.yaml')

# accessing in two different ways
dl = DataLoader(dataset, batch_size=config.hyperparameters.batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=config['hyperparameters']['lr'])
```
or using `hydra` for loading the configuration
```python
import hydra

@hydra.main(config_name="basic.yaml")
def main(cfg):
    print(cfg.hyperparameters.batch_size, cfg.hyperparameters.learning_rate)

if __name__ == "__main__":
    main()
```
The idea behind refactoring our hyperparameters into `.yaml` files is that we disentangle the model configuration from the model. In this way it is easier to do version control of the configuration because we have it in a seperate file.

### Exercises

The main idea behind the exercises is to take a single script (that we provide) and use Hydra to make sure that everything gets correctly logged such that you would be able to exactly report to others how each experiment was configured. In the provided script, the hyperparameters are hardcoded into the code and your job will be to separate them out into a configuration file.

Note that we provide an solution (in the `vae_solution` folder) that can help you get through the exercise, but try to look online for your answers before looking at the solution. Remember: its not about the result, its about the journey.

1. Start by install hydra: `pip install hydra-core --upgrade`

2. Next take a look at the `vae_mnist.py` and `model.py` file and understand what is going on. It is a model we will revisit during the course.
   
3. Identify the key hyperparameters of the script. Some of them should be easy to find, but at least 3 have made it into the core part of the code. One essential hyperparameter is also not included in the script but is needed to be completely reproducible (HINT: the weights of any neural network is initialized at random).
   
4. Write a configuration file `config.yaml` where you write down the hyperparameters that you have found

5. Get the script running by loading the configuration file inside your script (using hydra) that incorporates the hyperparameters into the script. Note: you should only edit the `vae_mnist.py` file and not the `model.py` file.
   
6. Run the script

7. By default hydra will write the results to a `outputs` folder, with a sub-folder for the day the experiment was run and further the time it was started. Inspect your run by going over each file the hydra has generated and check the information has been logged. Can you find the hyperparameters?
   
8. Hydra also allows for dynamically changing and adding parameters on the fly from the command-line:

   1. Try changing one parameter from the command-line
      ```bash
      python vae_mnist.py seed=1234
      ```

   2. Try adding one parameter from the command-line
      ```bash
      python vae_mnist.py +experiment.stuff_that_i_want_to_add=42
      ```

9. By default the file `vae_mnist.log` should be empty, meaning that whatever you printed to the terminal did not get picked up by Hydra. This is due to Hydra under the hood making use of the native python [logging](https://docs.python.org/3/library/logging.html) package. This means that to also save all printed output from the script we need to convert all calls to `print` with `log.info`

   1. Create a logger in the script:
      ```python
      import logging
      log = logging.getLogger(__name__)
      ```

   2. Exchange all calls to `print` with calls to `log.info`

   3. Try re-running the script and make sure that the output printed to the terminal also gets saved to the `vae_mnist.log` file

10. Make sure that your script is fully reproducible. To check this you will need two runs of the script to compare. Then run the `reproduceability_tester.py` script as
    ```bash
    python reproducibility_tester.py path/to/run/1 path/to/run/2
    ```
    the script will go over trained weights to see if the match and that the hyperparameters was the same. Note: for the script to work, the weights should be saved to a file called `trained_model.pt` (this is the default of the `vae_mnist.py` script, so only relevant if you have changed the saving of the weights)

11. Finally, make a new experiment using a new configuration file where you have changed a hyperparameter of your own choice. You are not allowed to change the configuration file in the script but should instead be able to provide it as an argument when launching the script e.g. something like
    ```bash
    python vae_mnist.py experiment=exp2
    ```
  
### Final exercise

Make your MNIST code reproducible! Apply what you have just done to the simple script to your MNIST code. Only requirement is that you this time use multiple configuration files, meaning that you should have at least one `model_conf.yaml` file and a `training_conf.yaml` file that separates out the hyperparameters that have to do with the model definition and those that have to do with the training.