# Obico Evaluation
The directory contains the code to evaluate the Obico Spaghetti Detective model on the test sets. All code is adapted from the original repository which can be found [here](https://github.com/TheSpaghettiDetective/obico-server).

## Setup
1. Install the required packages:
```bash
pip install -r requirements.txt
```

2. Download the model weights from [this link](https://tsd-pub-static.s3.amazonaws.com/ml-models/model-weights-5a6b1be1fa.onnx) and place it in the `model` directory. Optionally, you can directly download the model weights using the following command:
```bash
wget https://tsd-pub-static.s3.amazonaws.com/ml-models/model-weights-5a6b1be1fa.onnx -O model/model-weights-5a6b1be1fa.onnx
```
3. Ensure all required datasets are downloaded and placed in the root 'data' directory. Details on how to download and prepare the datasets can be found [here](../README.md).

4. Run the evaluation script:
```bash
python main.py
```