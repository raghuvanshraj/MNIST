import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
        self.conv2 = nn.Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)

        x = x.view(-1, 4 * 4 * 50)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def predict(self, x: torch.Tensor) -> int:
        with torch.no_grad():
            x = self.forward(x)
            pred = torch.argmax(x, dim=1)

            return pred.data
