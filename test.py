import torch
from model import RamanNN
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('weights')
args = parser.parse_args()

df = pd.read_csv('spectra/test_data.csv')
test_data = torch.tensor(df.iloc[:, 1:].to_numpy(), dtype=torch.float32)
test_data = torch.unsqueeze(test_data, 1)

model = RamanNN()
model.load_state_dict(torch.load(args.weights, weights_only=True))
model.eval()

pred = model(test_data).argmax(1).numpy()
df = pd.DataFrame(pred, columns=['label'])
df.index.name = 'id'
df.to_csv('results.csv')
