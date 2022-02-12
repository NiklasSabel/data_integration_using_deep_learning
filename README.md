
Data Integration using Deep Learning
==============================

Data Integration using Deep Learning
This github-repository provides the code for our Team Project in the fall semester 2021 at the University of Mannheim with the topic of Data Integration using Deep Learning. We will try to investigate how Transformer Models like BERT work on tasks in data integration with special focus on entity and schema matching.

The team consists of Jennifer Hahn, Jannik Reißfelder,  Kim-Carolin Lindner, Niklas Sabel, Cheng Chen, Marvin Rösel, Estelle Weinstock and Luisa Theobald.



Table of Contents
==============================

.. contents::

Description
==============================

We make all our code availabe that were used for this project. It contains the data preprocessing, inducing cross-lingual word embeddings, training and evaluating all models. You can find the code for each part in the following table: 

*  `Data Preprocessing <https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval/tree/main/src/data/>`__
*  `Feature Generation <https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval/tree/main/src/features>`__
*  `Inducing CLWE <https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval/tree/main/src/embeddings>`__
*  `Training and Evaluating Supervised Models <https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval/tree/main/src/models>`__

All Experiments done were written in Jupyter Notebooks, which can be found in this `Folder https://github.com/NiklasSabel/data_integration_using_deep_learning/tree/main/notebooks`__

Furthermore, we make all models available `Drive <https://drive.google.com/drive/folders/1r0UExZMI46dbYx_zfdVCmbPNJC3O8yU9?usp=sharing/>`__. All raw and preprocessed data can be downloaded in the following `Drive <https://drive.google.com/drive/folders/1EuDDZSmv2DWgw3itdGSDwKz3UYIcLVmT?usp=sharing/>`__. 


How to Install
##############

To use this code you have to follow these steps:

1. Start by cloning this Git repository:

.. code-block::

    $  git clone https://github.com/J4K08L4N63N84HN/crosslingual-information-retrieval.git
    $  cd crosslingual-information-retrieval

2. Continue by creating a new conda environment (Python 3.8):

.. code-block::

    $  conda create -n crosslingual-information-retrieval python=3.8
    $  conda activate crosslingual-information-retrieval

3. Install the dependencies:

.. code-block::

    $ pip install -r requirements.txt

For a detailed documentation you can refere to `here <https://crosslingual-information-retrieval.readthedocs.io/en/latest/index.html>`__ or create your own sphinx documentation with

Credits
#######

The project started in March 2021 as a Information Retrieval project at the University of Mannheim. The project team consists of:

* `Minh Duc Bui <https://github.com/MinhDucBui/>`__
* `Jakob Langenbahn <https://github.com/J4K08L4N63N84HN/>`__
* `Niklas Sabel <https://github.com/NiklasSabel/>`__

License
#######

This repository is licenced under the MIT License. If you have any enquiries concerning the use of our code, do not hesitate to contact us.

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
