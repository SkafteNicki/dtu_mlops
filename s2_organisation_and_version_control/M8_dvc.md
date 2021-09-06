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

In this module we are going to return to version control. However, this time we are going to focus on version control of data. The reason we need to seperate between standandard version control and data version control comes down to one problem: size. 

Classic version control was developed to keep track of code files, which all are simple text files. Even a codebase that contains 1000+ files with millon lines of codes can probably be stored in less than a single gigabyte (GB). On the
other hand, the size of data can be drastically bigger. As most machine learning algorithms only gets better with the
more data that you feed them, we are seeing models today that are being trained on petabytes of data (1.000.000 GB).