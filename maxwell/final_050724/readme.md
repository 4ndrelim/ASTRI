# Using a neural network to solve Maxwell equations

## Introduction
This project is designed to demonstrate the capabilities of a neural network model for solving maxwell equations. It includes scripts for training the model (`train.py`) and obtaining predictions with a trained model (`evaluate.py`).

It is based on this [paper](https://arxiv.org/pdf/1711.10561v1).

## Installation

This project is compatible with MacOs and Windows systems.

To install the required packages, run the following command:

```
pip install -r requirements.txt
```

If you do not have pip, see [here](https://pypi.org/project/pip/).


## Training
Refer to (`train.py`).

Before executing, 

### Dataset
1. The user should ensure `test.npy` and `train.npy` exists in the `dataset` directory
    - Note that the top of the file has a **USER CONFIG** section to customise the names of the datasets

### Training Parameters
2. The folowing arguments can be customised in the main portion (scroll to the end) of the script:
    - Optimizer
    - Scheduler
    - Epochs (by default, runs for 32)
    - Alpha (by default, set to 0.05)

### Saving the Model
3. Once complete, the produced model `my_model.pth` and its feature scaler `scaler.pkl` could be found in the directory `saved_model`
    - Names can be customised in the **USER_CONFIG**
    - If saving the not required, simply set `MODEL_PATH` and `SCALER_PATH` to `None` in the **USER CONFIG** section
    - Note if the paths are specified, then existing models will be overwritten


### Feedback
The script will output the model's performance metrics per epoch in the terminal. 
If the R2 on the validation set of the current epoch is worse than best found previously, model will not be evaluated on the test set for this epoch.


## Obtaining Predictions
Refer to (`evalute.py`).

Before executing,

### Dataset
1. The user should specify the path to the dataset in `DATASET_PATH`

### Load saved models
2. If (`train.py`) is executed, the saved model and scaler should already exist in the default created directory
    - Otherwise, make sure they exist and configure the path accordingly

