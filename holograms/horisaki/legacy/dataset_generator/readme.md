# Generating a dataset of both random and radial patterns

## Introduction

This project is designed to generate a numpy dataset of random and radial patterns using the script `dataset_generator.py` in order to train a neural network mode.


## Installation

This project is compatible with MacOs and Windows systems.

To install the required packages, run the following command:

```

pip install -r requirements.txt

```

If you do not have pip, see [here](https://pypi.org/project/pip/).


## Generating the dataset

A dataset containing both random and radial patterns would be created in the form of a numpy array.

    - Each item in the numpy array represents an image of `IMAGE_SIZE` x `IMAGE_SIZE*2` 

    - The user should adjust parameters in **USER_CONFIG**

    - The dataset is shuffled by default unless specified


## Saving the dataset

Once complete, the produced dataset `dataset.npy` could be found in the main directory`

    - Names can be customised in the **USER_CONFIG**

    - Note existing datasets with same name and format will be overwritten