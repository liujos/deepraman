
### DeepRaman

A 1D CNN inspired by LeNet-5 to classify bacterial strains based on their Raman Spectra.

## Usage

1. Download [data](https://www.kaggle.com/competitions/ramanspec/overview) into deepraman/spectra

Train:

 ```sh 
python3 train.py
```

Test:

```sh
python3 test.py weights/weight_x
```

2. Model predictions are output in results.csv

