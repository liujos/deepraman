import torch

class RamanNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 3, kernel_size=5, padding=2), # input: 1x1000, output: 3 @ 1x1000
            torch.nn.ReLU(), 
            torch.nn.MaxPool1d(kernel_size=2, stride=2), # input: 3 @ 1x1000, output: 3 @ 1x500
            torch.nn.Conv1d(3, 5, kernel_size=5), # input: 3 @ 1x500, output 5 @ 1x496
            torch.nn.ReLU(),
            torch.nn.MaxPool1d(kernel_size=2, stride=2), #input: 5 @ 1x496, output: 5 @ 1x248
        )

        self.fc = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(1240, 400),
            torch.nn.ReLU(),
            torch.nn.Linear(400, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 31),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)
