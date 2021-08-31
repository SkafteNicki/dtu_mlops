---
layout: default
title: M11 - Experiment logging
parent: S3 - Debugging, Profiling and Logging
nav_order: 3
---

# Title
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




### Experiment visualizers

While logging loss values to terminal, or plotting training curves in matplotlib may be enough doing smaller experiment,
there is no way around using a proper experiment tracker and visualizer when doing large scale experiments.

For these exercises we will initially be looking at incorporating [tensorboard](https://www.tensorflow.org/tensorboard) into our code, 
as it comes with native support in PyTorch

1. Install tensorboard (does not require you to install tensorflow)
   ```pip install tensorboard```

2. Take a look at this [tutorial](https://pytorch.org/docs/stable/tensorboard.html)

3. Implement the summarywriter in your training script from the last session. The summarywrite should log both
   a scalar (`writer.add_scalar`) (atleast one) and a histogram (`writer.add_histogram`). Additionally, try log
   the computational graph (`writer.add_graph`).
   
4. Start tensorboard in a terminal
   ```tensorboard --logdir this/is/the/dir/tensorboard/logged/to```
   
5. Inspect what was logged in tensorboard

Experiment visualizers are especially useful for comparing values across training runs. Multiple runs often
stems from playing around with the hyperparameters of your model.

6. In your training script make sure the hyperparameters are saved to tensorboard (`writer.add_hparams`)

7. Run at least two models with different hyperparameters, open them both at the same time in tensorboard
   Hint: to open multiple experiments in the same tensorboard they either have to share a root folder e.g.
   `experiments/experiment_1` and `experiments/experiment_2` you can start tensorboard as
   ```tensorboard --logdir experiments```
   or as
   ```tensorboard --logdir=name1:/path/to/logs/1,name2:/path/to/logs/2```

While tensorboard is a great logger for many things, more advanced loggers may be more suitable. For the remaining 
of the exercises we will try to look at the [wandb](https://wandb.ai/site) logger. The great benefit of using wandb
over tensorboard is that it was build with colllaboration in mind (whereas tensorboard somewhat got it along the
way).

1. Start by creating an account at [wandb](https://wandb.ai/site). I recommend using your github account but feel
   free to choose what you want. When you are logged in you should get an API key of length 40. Copy this for later
   use (HINT: if you forgot to copy the API key, you can find it under settings).

2. Next install wandb on your laptop
   ```
   pip install wandb
   ```

3. Now connect to your wandb account
   ```
   wandb login
   ```
   you will be asked to provide the 40 length API key. The connection should be remain open to the wandb server
   even when you close the terminal, such that you do not have to login each time. If using `wandb` in a notebook 
   you need to manually close the connection using `wandb.finish()`.

4. With it all setup we are now ready to incorporate `wandb` into our code. The interface is fairly simple, and
   this [guide](https://docs.wandb.ai/guides/integrations/pytorch) should give enough hints to get you through
   the exercise. (HINT: the two methods you need to call are `wandb.init` and `wandb.log`). To start with, logging
   the training loss of your model will be enough.

5. After running your model, checkout the webpage. Hopefully you should be able to see at least 

6. Now log something else than scalar values. This could be a image, a histogram or a matplotlib figure. In all
   cases the logging is still going to use `wandb.log` but you need extra calls to `wandb.Image` ect. depending
   on what you choose to log.

7. Finally, lets create a report that you can share. Click the **Create report** button where you choose the *blank*
   option. Then choose to include everything in the report 

8. To make sure that you have completed todays exercises, make the report shareable by clicking the *Share* button
   and create *view-only-link*. Send the link to my email `nsde@dtu.dk`, so I can checkout your awesome work.

9. Feel free to experiment more with `wandb` as it is a great tool for logging, organizing and sharing experiments.
