---
layout: default
title: M12 - Profiling
parent: S4 - Debugging, Profiling and Logging
nav_order: 2
---

# Profilers
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

## Profilers

In general profiling code is about improving the performance of your code. In this session we are going to take a somewhat narrow approach to what "performance" is: runtime, meaning the time it takes to execute your program. 

At the bare minimum, the two questions a proper profiling of your program should be able to answer is:

* *“How many times is each method in my code called?”*
* *“How long do each of these methods take?”*

The first question is important to priorities optimization. If two methods `A` and `B` have approximately the same runtime, but `A` is called 1000 more times than `B` we should probably spend time optimizing `A` over `B` if we want to speedup our code. The second question is gives itself, directly telling us which methods are the expensive to call.

Using profilers can help you find bottlenecks in your code. In this exercise we will look at two different
profilers, with the first one being the [cProfile](https://docs.python.org/3/library/profile.html), pythons
build in profiler.

### Exercises

1. Run the `cProfile` on the `vae_mnist_working.py` script. Hint: you can directly call the profiler on a
   script using the `-m` arg
   `python -m cProfile -o <output_file> -s <sort_order> myscript.py`
   
2. Try looking at the output of the profiling. Can you figure out which function took the longest to run?

3. Can you explain the difference between `tottime` and `cumtime`? Under what circumstances does these differ and when are they equal.

4. To get a better feeling of the profiled result we can try to visualize it. Python does not
   provide a native solution, but open-source solutions such as [snakeviz](https://jiffyclub.github.io/snakeviz/)
   exist. Try installing `snakeviz` and load a profiled run into it (HINT: snakeviz expect the run to have the file
   format `.prof`).

5. Try optimizing the run! (Hint: The data is not stored as torch tensor). After optimizing the code make sure 
   (using `cProfile` and `snakeviz`) that the code actually runs faster.

## Pytorch profiling

Profiling machine learning code can become much more complex because we are suddenly beginning to mix different 
devices (CPU+GPU), that can (and should) overlap some of their computations. When profiling this kind of machine 
learning code we are often looking for *bottlenecks*. A bottleneck is simple the place in your code that is 
preventing other processes from performing their best. This is the reason that all major deep learning 
frameworks also include their own profilers that can help profiling more complex applications.

The image below show a typical report using the 
[build in profiler in pytorch](https://www.google.com/search?client=firefox-b-d&q=pytorch+profiling). 
As the image shows the profiler looks both a the `kernel` time (this is the time spend doing actual computations) 
and also transfer times such as `memcpy` (where we are copying data between devices). 
It can even analyze your code and give recommendations.

<p align="center">
  <img src="../figures/pytorch_profiler.png" width="700" title="hover text">
</p>

Using the profiler can be as simple as wrapping the code that you want to profile with the `torch.profiler.profile` decorator

```python
with torch.profiler.profile(...) as prof:
   # code that I want to profile
   output = model(data)
```

### Exercises (optional)

In these investigate the profiler that is build into PyTorch already. Note that these exercises requires that you 
have PyTorch v1.8.1 installed (or higher). You can always check which version you currently have installed by writing 
(in a python interpreter):

```python
import torch
print(torch.__version__)
```

Additionally, to display the result nicely (like `snakeviz` for `cProfile`) we are also going to use the 
tensorboard profiler extension

```bash 
pip install torch_tb_profiler
```

1. The documentation on the new profiler is sparse but take a look at this
   [blogpost](https://pytorch.org/blog/introducing-pytorch-profiler-the-new-and-improved-performance-tool/)
   and the [documentation](https://pytorch.org/docs/stable/profiler.html) which should give you an idea of 
   how to use the PyTorch profiler.

2. Lets try out an simple example:

   1. Try to run the following code
      ```python
      import torch
      import torchvision.models as models
      from torch.profiler import profile, record_function, ProfilerActivity

      model = models.resnet18()
      inputs = torch.randn(5, 3, 224, 224)

      with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
         with record_function("model_inference"):
            model(inputs)
      ```
      this will profile the `forward` pass of resnet 18 model. 
      
   2. Running this code will produce an `prof` object that contains all the relevant information about the profiling. 
      Try writing the following code:
      ```python
      print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
      ```
      what operation is taking most of the cpu?

   3. Try running
      ```python
      print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
      ```
      can you see any correlation between the shape of the input and the cost of the operation?

   4. (Optional) If you have a GPU you can also profile the operations on that device:
      ```python
      with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
         with record_function("model_inference"):
            model(inputs)
      ```

3. The `torch.profiler.profile` function takes some additional arguments. What argument would you need to 
   set to also profile the memory usage? (Hint: this [page](https://pytorch.org/docs/stable/profiler.html))
   Try doing it to the simple example above and make sure to sort the sample by `self_cpu_memory_usage`.

4. As mentioned we can also get a graphical output for better inspection. After having done a profiling
   try to export the results with:
   ```python
   prof.export_chrome_trace("trace.json")
   ```
   you should be able to visualize the file by going to `chrome://tracing` in any chromium based web browser.

5. Additionally, we can also vizualize the profiling results using the profiling viewer in tensorboard. Simply
   initialize the `profile` function with an additional argument:
   ```python
   from torch.profiler import profile, tensorboard_trace_handler
   with profile(..., on_trace_ready=tensorboard_trace_handler(<profile_dir>)):
      ...
   ```
   where `<profile_dir>` you choose yourself. After doing a profile you should see a file being created in the
   chosen folder having the file extension `.pt.trace.json`. Finally, launch tensorboard and look a the profiled
   result
   ```bash
   tensorboard --logdir <profile_dir>
   ```
   
6. Redo the steps above on the `vae_mnist_working.py` file, implementing now multiple calls to `record_function`
   on various levels of the training. Try running the profiling and investigate if you are able to improve the code.
