---
layout: default
title: M18 - Using the cloud
parent: S6 - The cloud
nav_order: 2
---

# Using the cloud
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

In this set of exercises we are going to get more familiar with the using some of the resources that google cloud offers.

## Compute


### Exercises

We are now going to start actually using the cloud.

1. Enable the `Compute Engine API`. You should be able to find it in the sidebar.

2. Try to `Create instance`. You will see the following image below.
   <p align="center">
     <img src="../figures/gcp4.png" width="800" title="hover text">
   </p>
   Give it a meaningful name, set the location to some location that is closer to where you actually is (to reduce latency). Finally try to adjust the the configuration a bit. What two factors are effecting the price of the compute unit? 
   
3. After figuring this out, create a `e2-medium` instance (leave rest configured as default). Before clicking the `Create` button make sure to check the `Equavalent Command Line` botton. You should see a very long command that you could have typed instead to do the exact same.

4. Now in a local terminal type:
   ```bash
   gcloud compute instances list
   ```
   you should hopefully see the instance you have just created.

5. You can start a terminal directly by typing:
   ```bash
   gcloud beta compute ssh --zone <zone> <name> --project <project-name> 
   ```

6. While logged into the instance you will see that neither python or pytorch is installed. 

```
gcloud compute instances create %INSTANCE_NAME% \
  --zone=%ZONE% 
  --image-family=%IMAGE_FAMILY% 
  --image-project=deeplearning-platform-release
```

## Data storage
Another big part of cloud computing is storage of data. There are many reason that you want to store your data in the cloud including:

- Easily being able to share
- Easily expand as you need more
- Data is stored multiple locations, making sure that it is not lost in case of an emergency

Cloud storage is luckily also very cheap. Google cloud only takes around $0.026 per GB per month. This means that around 1 TB of data would cost you $26 which is more than what the same amount of data would cost on Goggle Drive, but the storage in Google cloud is much more focused on enterprise where you have a need for accessing data through an API.

### Exercises
When we did the exercise on data version control, we made `dvc` work together with our own google drive to storage data. However, a big limitation of this is that we need to authentic each time we try to either push or pull the data. The reason is that we need to use an API instead which is offered through google cloud.

We are going to follow the instructions from this [page](https://dvc.org/doc/user-guide/setup-google-drive-remote)

1. Lets start by creating a data storage. On the GCP startpage, in the sidebar, click on the `Cloud Storage`. On the next page click the `Create bucket`:
   <p align="center">
     <img src="../figures/gcp5.png" width="800" title="hover text">
   </p>
   Give the bucket an unique name, set it to a region close by and make it of size 20 GB as seen in the image.

2. After creating the storage, you should be able to see it if you type
   ```bash
   gsutil ls
   ```
   `gsutil` is an additional command to `gcloud`, that provides more command line options.

2. Next we need the google storage extension for `dvc`
   ```bash
   pip install dvc[gs]
   ```

3. Now in your mnist reposatory where you have already configured dvc, we are going to change the storage from our google drive to our newly created google cloud storage.
   ```bash
   dvc remote add -d remote_storage <output-from-gsutils>
   ```

4. The above command will change the `.dvc/config` file. Add and commit that file. Finally, push data to the cloud
   ```bash
   dvc push
   ```

### Container registry

We are now going to return to docker. We have until now seen how we can automitize building images using github actions. Now, we are going to automize the
process of uploading the build containers to a so called `container registry.

1. At the homepage of *gcp* type in `Google Container Registry API` in the search bar. Find the service and enable it.

