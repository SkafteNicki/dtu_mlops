---
layout: default
title: M24 - Data drifting
parent: S8 - Deployment
nav_order: 3
---

# Data drifting
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


## Drifting data

A big part of deployment is also monitoring the performance of your deployed model. Especially, the concept of
detecting **data drifting** comes into play. Data drift is one of the top reasons model accuracy degrades over time. 
For machine learning models, data drift is the change in model input data that leads to model performance degradation. 
In practical terms this means that the model is receiving input that is outside of the scope that it was trained on. 

For this exercise we will rely on **TorchDrift** that contains state-of-the-art methods for detecting data
drifting in classification models. It is a model agnostic tool that should work with any black-box classification
model (in Pytorch). You can learn more about TorchDrift in this [video](https://www.youtube.com/watch?v=rV5BhoKILoE&t=1s).

### Exercises

1. Install TorchDrift `pip install torchdrift`

2. Look over the following [example](https://torchdrift.org/notebooks/drift_detection_on_images.html). It goes
   over the API of TorchDrift and better explains the concept of drifting data distributions
   
3. Implement drift detection on your MNIST example, this includes

    3.1 Start by creating artificially drifted MNIST data similar to the above example (HINT: apply Gaussian
        blur to the training data)
        
    3.2 Implement drift detection by going over the same steps as in the example. You can skip over a large
        part of it because you should already have a trained classifier ready to rock.
    
    3.4 While the `p-value` calculated by the drift detectors are great for having a single number for detecting
        if input is within spec, while developing it is beneficially to visualize the training data distribution
        and the drifting distribution. However, as the features we are visualizing are higher dimension than
        2, we need to resort to dimensionality reduction. In the example they use IsoMap. Try out other
        dimensionality reduction methods (HINT: [sklearn](https://scikit-learn.org/stable/modules/manifold.html))
    
    3.4 TorchDrift comes with multiple drift detectors and one may work better on your dataset than others. 
        Investigate the different choices of `kernel` parameter for the `KernelMMDDriftDetector`. You are
        supposed to produce one table that compares the `score` and `p-value` for the different detectors and a single
        plot with a subplot of embeddings of each detector similar to the figure below
    
    3.5 Repeat the exercise with some out-of-distribution data for example 
        [FashionMnist](https://github.com/zalandoresearch/fashion-mnist) dataset. Is it easier for the
        drift detectors to figure out true out-of-distribution data compared to the slightly blurred data?

![exercise](../figures/drifting_ex.PNG)