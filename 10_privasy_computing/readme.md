# 10. Privacy computing


## Continues-Integration

Continues integration (CI) is a development practise that makes sure that updates to code 
are automatically tested such that it does not break existing code. When we look at MLOps,
CI belongs to the operation part. 

It should be notes that applying CI does not magically secure that your code does not break.
CI is only as strong as the unittest that are automatically executed. CI simply structures and
automates this.

<p align="center">
<b> “Continuous Integration doesn’t get rid of bugs, but it does make them dramatically easier to find and remove.” -Martin Fowler, Chief Scientist, ThoughtWorks </b>
![ci](../figures/ci.png)
</p>



### 
Continuous Integration (CI) is a development practice where developers integrate code into a shared repository frequently, preferably several times a day. Each integration can then be verified by an automated build and automated tests. While automated testing is not strictly part of CI it is typically implied.

One of the key benefits of integrating regularly is that you can detect errors quickly and locate them more easily. As each change introduced is typically small, pinpointing the specific change that introduced a defect can be done quickly.

In recent years CI has become a best practice for software development and is guided by a set of key principles. Among them are revision control, build automation and automated testing.

Additionally, continuous deployment and continuous delivery have developed as best-practices for keeping your application deployable at any point or even pushing your main codebase automatically into production whenever new changes are brought into it. This allows your team to move fast while keeping high quality standards that can be checked automatically.




