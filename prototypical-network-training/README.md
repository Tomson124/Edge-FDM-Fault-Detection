# Prototypical Network Training and Evaluation
This repository contains code for training and evaluating Prototypical Networks for few-shot learning tasks. The implementation is based on the original paper "_Prototypical Networks for Few-shot Learning_" by Snell et al. (2017) and adapts the code provided by the authors which is avaliable [here](https://github.com/jakesnell/prototypical-networks).

## Setup
1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Install the protonets package by running `python setup.py install` or `python setup.py develop`.

3. Download the datasets for training and evaluation. Instructions for downloading and preparing the datasets used for training the 3D fault detection model can be found [here](../README.md).

## Training
Training of the network can be carried out using the `scripts/train/few_shot/run_train.py` script. The script takes several arguments to customize the training process. The default values are used to train the most performant model for the 3D fault detection task. Refer to the script for more details on the available arguments.

## Evaluation
### Accuracy, Loss, recall and precision (F1 score)
The evaluation of the trained model can be done using the `scripts/predict/few_shot/run_eval.py` script. The script takes several arguments to customize the evaluation process. The default values are used to evaluate the most performant model for the 3D fault detection task. Refer to the script for more details on the available arguments.

### Image Throughput
Image throughput is calculated by running the `scripts/predict/few_shot/run_benchmark.py`, where the default values are used to evaluate the most performant model for the 3D fault detection task. Refer to the script for more details on the available arguments.