
Data Integration using Deep Learning
==============================

Data Integration using Deep Learning
This github-repository provides the code for our Team Project in the fall semester 2021 at the University of Mannheim with the topic of Data Integration using Deep Learning. We are investigating the performance of table transformer frameworks on the tasks of matching entities (entity matching) and schemes (schema matching) across tables. The task will be presented as a multi-class classification problem of characterising if an row (entity) or a column (schema) belongs to a predefined cluster/has a specific label. Table transformer models use whole table representations as input to enrich the classification problem with metadata and surrounding information. We investigate whether these enriched models, in particular TURL and Tabbie, will perform better than our two baselines: RandomForest based on tf-idf and word count representations and RoBERTa/TinyBert as embedding based transformer models.


Table of Contents
==============================

.. contents::

Description Entity
==============================

We make all our code availabe that were used for this project. In the following we will shortly describe the project setup and our approach.

You can find the code for each part in the following table: 

*  `Data set generation >`__
*  `Baselines <>`__
*  `TURL Experiments <>`__
*  `Tabbie Experiments <>`__
*  `Visualizations <https://github.com/NiklasSabel/data_integration_using_deep_learning/tree/main/visualizations>`__

All Experiments done were written in Jupyter Notebooks, which can be found in this  `Folder <https://github.com/NiklasSabel/data_integration_using_deep_learning/tree/main/notebooks/Entity>`__

Description Schema
==============================

We make all our code availabe that were used for this project. It contains the . You can find the code for each part in the following table: 

*  `Data set generation >`__
*  `Baselines <>`__
*  `TURL Experiments <>`__
*  `Tabbie Experiments <>`__
*  `Visualizations <>`__

All Experiments done were written in Jupyter Notebooks, which can be found in this  `Folder <https://github.com/NiklasSabel/data_integration_using_deep_learning/tree/main/notebooks/Schema>`__

Furthermore, we make all models available `Drive <url>`__. All raw and preprocessed data can be downloaded in the following `Drive <url>`__. 


How to Install
==============================

To use this code you have to follow these steps:

1. Start by cloning this Git repository:

.. code-block::

    $  git clone https://github.com/NiklasSabel/data_integration_using_deep_learning.git
    $  cd data_integration_using_deep_learning

2. Continue by creating a new conda environment (Python 3.8):

.. code-block::

    $  conda create -n data_integration_using_deep_learning python=3.8
    $  conda activate crosslingual-information-retrieval

3. Install the dependencies:

.. code-block::

    $ pip install -r requirements.txt

Credits
==============================

The project started in October 2021 as a team project at the University of Mannheim and ended in March 2022. The project team consists of:

* `Cheng Chen <https://github.com/chengc823>`__
* `Jennifer Hahn <https://github.com/JenniferHahn>`__
* `Kim-Carolin Lindner <https://github.com/kimlindner>`__
* `Jannik Reißfelder <https://github.com/jannik-reissfelder>`__
* `Marvin Rösel <https://github.com/maroesel>`__
* `Niklas Sabel <https://github.com/NiklasSabel/>`__
* `Luisa Theobald <https://github.com/LuThe17>`__
* `Estelle Weinstock <https://github.com/estelleweinstock>`__



License
==============================

This repository is licenced under the MIT License. If you have any enquiries concerning the use of our code, do not hesitate to contact us.

Project based on the  `cookiecutter data science project template <https://drivendata.github.io/cookiecutter-data-science/>`__ #cookiecutterdatascience

`TURL repository <https://github.com/sunlab-osu/TURL>`__

`Tabbie repository <https://github.com/SFIG611/tabbie>`__

