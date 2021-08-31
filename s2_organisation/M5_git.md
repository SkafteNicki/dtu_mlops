---
layout: default
title: M5 - Git
parent: S2 - Organisation
nav_order: 1
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

## Git 

Proper collaboration with other people will require that you can work on the same codebase in a organized manner.
This is the reason that **version control** exist. Simply stated, it is a way to keep track of:

* Who made changes to the code
* When did the change happen
* What changes where made

For a full explanation please see this [page](https://git-scm.com/book/en/v2/Getting-Started-What-is-Git%3F)

Secondly, it is important to note that Github is not git! Github is the dominating player when it comes to
hosting repositories but that does not mean that they are the only once (see [bitbucket](https://bitbucket.org/product/) 
for another example).

That said we will be using git+github throughout this course. It is a requirement for passing this course that 
you create a public repository with your code and use git to upload any code changes. How much you choose to
integrate this into your own projects depends, but you are at least expected to be familiar with git+github.

### Exercise

1. Install git on your computer and make sure that your installation is working by writing `git help` in a 
   terminal and it should show you the help message for git.

2. Create a [github](github.com/) account 

3. In your account create an repository, where the intention is that you upload the code from the final exercise
   from yesterday
   
   3.1 After creating the repository, clone it to your computer
       ```git clone https://github.com/my_user_name/my_repository_name.git```
       
   3.2 Move/copy the three files from yesterday into the repository
   
   3.3 Add the files to a commit by using `git add` command
   
   3.4 Commit the files using `git commit`
   
   3.5 Finally push the files to your repository using `git push`. Make sure to check online that the files
       have been updated in your repository.

4. If you do not already have a cloned version of this repository, make sure to make one! I am continuously updating/
   changing some of the material and I therefore recommend that you each day before the lecture do a `git pull` on your
   local copy

5. Git may seems like a waste of time when solutions like dropbox, google drive ect exist, and it is
   not completely untrue when you are only one or two working on a project. However, these file management 
   systems falls short when we hundred to thousand of people work to together. For this exercise you will
   go through the steps of sending an open-source contribution:
   
   5.1 Go online and find a project you do not own, where you can improve the code. For simplicity you can
       just choose the repository belonging to the course. Now fork the project by clicking the *Fork* botton.
       ![forking](../figures/forking.PNG)
       This will create a local copy of the repository which you have complete writing access to. Note that
       code updates to the original repository does not update code in your local repository.

   5.2 Clone your local fork of the project using ```git clone```

   5.3 As default your local repository will be on the ```master branch``` (HINT: you can check this with the
       ```git status``` commando). It is good practise to make a new branch when working on some changes. Use
       the ```git branch``` command followed by the ```git checkout``` command to create a new branch.

   5.4 You are now ready to make changes to repository. Try to find something to improve (any spelling mistakes?).
       When you have made the changes, do the standard git cycle: ```add -> commit -> push```


   5.5 Go online to the original repository and go the ```Pull requests``` tab. Find ```compare``` botton and
       choose the to compare the ```master branch``` of the original repo with the branch that you just created
       in your own repo. Check the diff on the page to make sure that it contains the changes you have made.

   5.6 Write a bit about the changes you have made and click send :)
