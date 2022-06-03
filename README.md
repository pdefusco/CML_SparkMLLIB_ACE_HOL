# Credit Scoring with PySpark ML

## Project Summary

This tutorial demonstrates how you can develop a Credit Scoring Model with PySpark. The model is used to score customers at scale via Batch Scoring and REST API. 

The tutorial is divided into the following parts:

1. Exploring the data via PySpark
2. Experimenting with Oversampling at scale (this part of the demo will soon be recreated as soon as ML Flow is added to CML)
3. Creating a SparkML Pipeline with training data
4. Scoring customers in batch with CML Jobs
5. Scoring customers in near-real time via a REST API



## Prerequisites

This project requires access to a CML Workspace or a CDSW Cluster. CML could either be in CDP Public or Private Cloud. 

Some familiarity with Python, PySpark, general API concepts in general and Jupyter Notebooks is recommended. 
However, no coding is required beyond simply executing the provided notebooks.

If you are completely new to CML and would like a quick intro to creating Projects, Sessions, using Spark and more, 
please start with [this repository](https://github.com/pdefusco/CML_CrashCourse)


## CML Project Setup

Navigate to the CDP Management Console and open your environment.

![alt text](img/cml2cde_readme01.png)

In the next page, take note of the region. For example, in the screenshot below this is ```“us-east-2”```. 
Save this to your editor, you will need this later.

Navigate to your CML Workspace and create a new project as shown below. 

![alt text](img/cr_score_2.png)

Select the Git Option and paste the following URL:

```https://github.com/pdefusco/Credit_Scoring_SparkML.git```

There is no need to customize the Runtime Setup settings at the bottom of the page. Default option values are fine. 

Within the project, launch a CML Session with the following settings:

~~~
Session Name: Setup Session (anything is fine here)
Editor: Workbench
Kernel: Python 3.7 or above
Enable Spark: disabled
Resource Profile: 2 vCPU / 4 GiB Memory - 0 GPUs
Other settings: default values are fine.
~~~

![alt text](img/cml2cde_readme2.png)


Next, open script ```“0_Setup.py”``` and hit the play button at the top of the screen to run all code at once. 

![alt text](img/cr_score_4.png)


This script will print the Cloud Storage environment variable to the screen, download all data for the tutorial, and move it from the local /home/cdsw folder to a new Cloud Storage folder.

This works for both AWS and Azure without edits.

Finally, go back to CML and kill your current Workbench Editor session. You won’t need it anymore.

![alt text](img/cml2cde_readme8.png)



## Part 1: Data Exploration with PySpark

Navigate back to the CML Project home. Launch a new session with the following settings:

~~~
Session Name: “JupyterLab Session”
Editor: JupyterLab
Kernel: Python 3.7 or higher
Enable Spark: Select a Spark 3.1+ Add-On
Resource Profile: 2 vCPU / 4 GiB Mem - No GPUs required.
Other options: you can leave the remaining options such as Edition and Version to their default values. 
~~~

![alt text](img/cr_score_9.png)

On the left, double click on the "1_Data_Exploration.ipynb" notebook to open it. You will execute the code in each cell to familiarize yourself with the data.

The notebook includes instructions and comments that explain what each part of the code. No code changes are required. Just execute each cell by either pressing the "play button" at the top, or highlighting it and entering "Shift" + "Enter" on your keyboard. Notice each cell requires a bit of time to run.

![alt text](img/cr_score_10.png)


## Part 2: Oversampling Experiments with PySpark