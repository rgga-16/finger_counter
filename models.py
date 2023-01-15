import torch 
import torch.nn as nn
import torch.nn.functional as F

class CustomClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.conv4 = nn.Conv2d(32, 32, 5)
        self.fc1 = nn.Linear(32*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84,1)
        self.fc4 = nn.Linear(84,6)

    def forward(self, x):
        # x(4,3,128,128)
        x = F.relu(self.conv1(x)) # x(4,3,128,128)==>x(4,6,124,124)
        x = self.pool(x) # x(4,6,124,124)==>x(4,6,62,62)
        x = F.relu(self.conv2(x)) # x(4,6,62,62)==>x(4,16,58,58)
        x = self.pool(x) # x(4,16,58,58)==>x(4,16,29,29)
        x = F.relu(self.conv3(x)) # x(4,16,29,29)==>x(4,32,25,25)
        x = self.pool(x) # x(4,32,25,25)==>x(4,32,12,12)
        x = F.relu(self.conv4(x)) # x(4,32,12,12)==>x(4,32,8,8)
        x = self.pool(x) # x(4,32,8,8)==>x(4,32,4,4)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x_handtypes = torch.sigmoid(self.fc3(x))
        x_counts = self.fc4(x)

        return x_handtypes,x_counts