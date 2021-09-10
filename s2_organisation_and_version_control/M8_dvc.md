---
layout: default
title: M8 - Data Version Control
parent: S2 - Organisation and version control
nav_order: 4
---

# Data Version Control
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

In this module we are going to return to version control. However, this time we are going to focus on version control of data. 
The reason we need to separate between standandard version control and data version control comes down to one problem: size. 

Classic version control was developed to keep track of code files, which all are simple text files. Even a codebase that 
contains 1000+ files with million lines of codes can probably be stored in less than a single gigabyte (GB). On the
other hand, the size of data can be drastically bigger. As most machine learning algorithms only gets better with the
more data that you feed them, we are seeing models today that are being trained on petabytes of data (1.000.000 GB).

Because this is a important concept there exist a couple of frameworks that have specialized in versioning data such as
[dvc](https://dvc.org/), [DAGsHub](https://dagshub.com/), [Hub](https://www.activeloop.ai/), [Modelstore](https://modelstore.readthedocs.io/en/latest/)
and [ModelDB](https://github.com/VertaAI/modeldb/). We are here going to use `dvc` as they also provide tools for
automatizing machine learning, which we are going to focus on later.

### DVC: What is it?

DVC (Data Version Control) is simply an extension of `git` to not only take versioning data but also models and experiments
in general. But how does it deal with these large data files? Essentially, `dvc` will just keep track of a small *metafile*
that will then point to some remote location where you original data is store. *metafiles* essentially works as placeholders
for your datafiles. Your large datafiles are then stored in some remote location such as Google drive or an `S3` bucket from
Amazon.






