# Training in the sky
Running computations locally is often sufficient when only playing around with code in
initial phase of development.

There exist a [numerous](https://github.com/zszazi/Deep-learning-in-cloud) cloud compute providers with
some of the biggest being:
* Azure
* AWS
* Alibaba cloud
* Google Cloud

In this course we are going to focus on Azura, the solution from Microsoft. It should be noted that
today exercises only will give a glimse of what Azura can provide.

### Exercises

1. Create an account at https://azure.microsoft.com/en-us/free/. If you already have an Azure account make
   sure to sign up with a new account so you are sure that you get the $200 free credit which is nessesary to
   complete the exercises
   
2. Login with your account and follow these instructions:
   https://docs.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources
   This will help you get started with Azure machine learning interface and how to start an compute
   instance. By the end of the exercise you should be able to start executing scripts in the cloud
   
3. Next follow these instructions:
   https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-hello-world
   This will go over how to launch your first script ("Hallo World") on the Azure platform.
   
4. Next follow these instructions
   https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-1st-experiment-sdk-train
   These will go over how to launch a training script in Azure.
   
5. With the basics in place, lets move on to a bit more advance. Close down the cpu compute instance
   that you was asked to create during the first instructions and now create a compute instance that
   have a GPU equipped.
   
6. Adjust the training script from the last instructions to run on GPU:

   6.1 Add `.to('cuda')` in appropiate places
   
   6.2 Running on GPU will require you to change the `config.run_config.environment` to a curated enviroment 
       that has cuda enabled pytorch installed. You can check which are availble by copy-and-running the
       `enviroments.py` script in Azure

7. We have provided a script called `???.py`. The objective is to get this running on Azure, but doing this
   naively will promt you that some packages are not installed because the default enviroment that we have
   been using until now only have some specific packages installed. Go over this 
   [page](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-use-environments) on how to
   create your own enviroment.
   
8. Finally we are going to get Azure running locally. By this we mean that instead of copying everything
   to the online webbrowser we are going to connect to Azure locally by using their API.

9. (Optional) After doing all of the above exercises can you figure out how many credit you have used.

10. Remember to stop whatever compute resourses that you have running to make sure that you do not burn
    through your credit on the first day ;)


https://docs.microsoft.com/en-us/cli/azure/install-azure-cli


### Final exercise

With the above completed exercises, try to get your MNIST code running on Azure. It does not have to
run for very long time