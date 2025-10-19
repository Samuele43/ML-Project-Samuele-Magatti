# MODEL.PY
#CNN definitions

import torch
import torch.nn as nn
import torch.nn.functional as F 
   

# MODEL FOR THE FIRST CNN

class TinyCNN(nn.Module):
        def __init__(self, num_classes=3):
            super(TinyCNN, self).__init__()
        
            # Solo due blocchi conv + pool
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)  # 3 -> 16 feature maps
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1) # 16 -> 32 feature maps
            self.pool = nn.MaxPool2d(2, 2)  # halves H e W in each step
        
            # compute dimensions for the fully connected layer
            # Input: 200x300 -> after 2 pool(2x2): 50x75
            self.fc1 = nn.Linear(32 * 50 * 75, 64)
            self.fc2 = nn.Linear(64, num_classes)
            self.dropout = nn.Dropout(0.3)


        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            # Flatten
            x = x.view(x.size(0), -1)
            # Fully connected
            x = self.dropout(F.relu(self.fc1(x)))
            x = self.fc2(x)
            return x
    



# MODEL FOR THE SECOND CNN

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        
        # covolutionals layers + ReLU + MaxPooling
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # input RGB -> 32 feature maps
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)  # halves H e W
        
        # compute final feature map size for the fully connected layer
        #  input images: 200x300 -> after 3 pool(2x2): 25x37 (approx)
        self.fc1 = nn.Linear(128 * 25 * 37, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization (to avoid overfitting) 
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Flatten

        x = x.view(x.size(0), -1)
        
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)  
        return x

    