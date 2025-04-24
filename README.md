# Few-shot learning for Real-Time Additive Manufacturing Fault Detection on Edge Devices
This repository contains code for training and evaluating Prototypical Networks (and fine-tuning feature extractors) as seen in the research paper "_Few-shot learning for Real-Time Additive Manufacturing Fault Detection on Edge Devices_" by Oliver Bravery.

## Quick Start
1. Ensure all datasets are downloaded and prepared. Instructions for downloading and preparing the datasets used for training the 3D fault detection model can be found [here](#dataset-preparation).

## Usage
- For fine-tuning the feature extractor, refer to [these instructions](./cnn-training/README.md).
- For training and evaluating the Prototypical Network, refer to [these instructions](./prototypical-network-training/README.md).
- For evaluating the Obico Spaghetti Detection Model, refer to [these instructions](./obico-evaluation/README.md).

## Dataset Preparation
The datasets used in the research 

1. Download the datasets from the following sources, extract them and placing them in `datasets/downloads`, naming them as follows:

| Dataset Name | Reference in paper | Download Link |
| ------------- | ------------- | ------------- |
| 1 | Projekt (2024) | https://universe.roboflow.com/automatisierung-projekt-aajut/3d-ho5at |
| 2 | QA (2022) | https://universe.roboflow.com/additive-manufacturing-qa/3dprinting |
| 3 | He (2022) | https://www.kaggle.com/datasets/mikulhe/3d-printing-errors |
| 4 | Bhardwaj (2025) | https://www.kaggle.com/datasets/bshaurya/3d-printing-success-failure-dataset-finetuned |
| 5 | Defects of 3D Printing (2024) | https://universe.roboflow.com/defects-of-3d-printing/fdm-sw9eb |
| 6 | Meftahmafazy (2023) | https://www.kaggle.com/datasets/muhammadmeftahmafazy/defect-3d-printing |
| 7 | Beyer (2024) | https://www.kaggle.com/datasets/nimbus200/3d-printing-errors |

2. Run the following script from the root directory to curate all datasets used in the research:
```bash
python datasets/map_datasets.py
```