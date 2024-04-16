
import torch
import torch.nn as nn
import torch.nn.functional as F 

class TinyNet(nn.Module):
    def __init__(self, num_classes=200):
        super(TinyNet, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1,stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
        )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
        )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3,padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv4 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        
        self.fc1 = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.LeakyReLU(),
        )
        
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers with batch normalization, ReLU activation, and max pooling
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # Flatten the tensor
        x = torch.flatten(x, 1)
        # Fully connected layers with ReLU activation
        x = self.fc1(x)
        x = self.fc2(x)
        return x




class TinyNet_drop(nn.Module):
    def __init__(self, num_classes=200):
        super(TinyNet_drop, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1,stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
        )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
        )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv4 = nn.Sequential(
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc1 = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
        )
        
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers with batch normalization, ReLU activation, and max pooling
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Flatten the tensor
        x = torch.flatten(x, 1)
        
        # Fully connected layers with ReLU activation
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x



class TinyNet_drop_residual(nn.Module):
    def __init__(self, num_classes=200):
        super(TinyNet_drop_residual, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Sequential(
                    nn.Conv2d(3, 32, kernel_size=3, padding=1,stride=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(),
                    nn.Conv2d(32, 32, kernel_size=3, padding=1,stride=2),
                    nn.BatchNorm2d(32),
                    nn.ReLU()
        )
        self.conv2 = nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, padding=1,stride=2),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
        )
        self.conv3 = nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.conv4 = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.conv5 = nn.Sequential(
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
        )

        self.downsample = nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc1 = nn.Sequential(
                    nn.Linear(4096, 512),
                    nn.ReLU(),
                    nn.Dropout(0.2),
        )
        
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # Convolutional layers with batch normalization, ReLU activation, and max pooling
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        residual = self.downsample(x)   
        #print("res:", residual.shape)
        x = self.conv4(x)
        x = self.conv5(x)
        #print(x.shape)

        x = residual + x

        x = F.relu(x)
        
        # Flatten the tensor
        x = torch.flatten(x, 1)
        #print(x.shape)
        # Fully connected layers with ReLU activation
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x