## About
DeepRaman is a deep learning model inspired by LeNet-5 to classify bacterial strains based on their Raman Spectra.

## Usage

To download the data:

1. Download the [data](https://www.kaggle.com/competitions/ramanspec/overview)
2. Move the train and test data into deepraman/spectra


To train the model:

1. Adjust any hyperparameters in train.py
2. 
```sh
python3 train.py
```

To test the model:
1. Pick the best model from weights/
2. 
```sh
python3 test.py weights/weight_x
```

3. Model predictions are output in results.csv

