

Here is an excellent
[research paper](https://www.microsoft.com/en-us/research/publication/to-blob-or-not-to-blob-large-object-storage-in-a-database-or-a-filesystem/) 
by microsoft that investigate if it is worth storing images or text (e.g. unstructured data in a classical database). The
conclusions are:

* If your pictures or document are typically below 256KB in size, storing them in a database VARBINARY column is more efficient
* If your pictures or document are typically over 1 MB in size, storing them in the filesystem is more efficient (and with SQL Server 2008's FILESTREAM attribute, they're still under transactional control and part of the database)
* In between those two, it's a bit of a toss-up depending on your use

If one decide to store images or text in a database, one should make sure to still keep it in its own database. This makes
sure that when you want to backup your database it becomes much easier, because there has been a seperation of concerns.

## Data systems



## ❔ Exercises

The database we are going to investigate here is [postgress](https://www.postgresql.org/) which is a very powerful
open-source relational database.

1. Start by installing sqlachemy

    ```bash
    pip install sqlachmy
    ```

2. Lets create a small script that ingest some data into a database

    1. Start by downloading either the green or yellow NYC taxi dataset. You can get them by either manually downloading
        from [this webpage](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) or alternatively you can run
        these two lines of code (assuming you have `wget` command installed (1)).
        { .annotate }
        
        1. :man_raising_hand: Windows users normally do not have `wget` installed. You can get a stand alone 
            installation of wget for Windows [here](https://gnuwin32.sourceforge.net/packages/wget.htm), or
            alternatively if you [chocolatey](https://community.chocolatey.org/) installed you can just run:

            ```bash
            choco install wget
            ```

        ```bash
        wget https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet
        wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2023-01.parquet
        ```

    2. To 




## 🧠 Knowledge check