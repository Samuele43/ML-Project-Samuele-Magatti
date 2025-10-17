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

# MODEL FOR THE THIRD CNN
#-----------------------------------------------
    #                  THIRD CNN
    #-----------------------------------------------



    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import time
    import copy
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
    import seaborn as sns
    import torchvision
    from torchvision.utils import make_grid
    import itertools
    import os
    import sys
    NUM_WORKERS = 0 if sys.platform == "darwin" else 4


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DEFINITIONS

    # CNN definitions

    class AdvancedCNN(nn.Module):
        def __init__(self, num_classes=3, dropout_prob=0.4):
            super().__init__()
            # deeper stack
            self.block1 = nn.Sequential(
                ConvBlock(3, 32),
                ConvBlock(32, 32, pool=False)
            )
            self.block2 = nn.Sequential(
                ConvBlock(32, 64),
                ConvBlock(64, 64, pool=False)
            )
            self.block3 = nn.Sequential(
                ConvBlock(64, 128),
                ConvBlock(128, 128, pool=False)
            )
            self.block4 = nn.Sequential(
                ConvBlock(128, 256),
                ConvBlock(256, 256, pool=False)
            )
            # reduce to global features
            self.global_pool = nn.AdaptiveAvgPool2d((1,1))  # output 256 x 1 x 1
            self.dropout = nn.Dropout(dropout_prob)
            self.fc = nn.Linear(256, num_classes)

        def forward(self, x):
            x = self.block1(x)   # /2
            x = self.block2(x)   # /4
            x = self.block3(x)   # /8
            x = self.block4(x)   # /16
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    class ConvBlock(nn.Module):
        def __init__(self, in_ch, out_ch, kernel=3, pool=True):
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2, bias=False)
            self.bn = nn.BatchNorm2d(out_ch)
            self.relu = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(2,2) if pool else nn.Identity()

        def forward(self, x):
            x = self.conv(x)
            x = self.bn(x)
            x = self.relu(x)
            x = self.pool(x)
            return x

    def train_one_epoch(model, loader, criterion, optimizer):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.detach().cpu().numpy())
        epoch_loss = running_loss / len(loader.dataset)
        acc = accuracy_score(all_targets, all_preds)
        return epoch_loss, acc

    def validate(model, loader, criterion):
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())
        epoch_loss = running_loss / len(loader.dataset)
        acc = accuracy_score(all_targets, all_preds)
        prec, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
        return epoch_loss, acc, prec, recall, f1, np.array(all_targets), np.array(all_preds)

    # functions to plot training curves

    # train and validation loss an accuracy

    def plot_training(hist):
        epochs = range(1, len(hist['train_loss'])+1)
        plt.figure(figsize=(12,4))
        plt.subplot(1,2,1)
        plt.plot(epochs, hist['train_loss'], label='train loss')
        plt.plot(epochs, hist['val_loss'], label='val loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.title('Loss')
        plt.subplot(1,2,2)
        plt.plot(epochs, hist['train_acc'], label='train acc')
        plt.plot(epochs, hist['val_acc'], label='val acc')
        plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.title('Accuracy')
        plt.tight_layout()
        plt.show()


    #plot confusion matrix    

    def plot_confusion(y_true, y_pred, classes):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, cmap='Blues')
        plt.ylabel('True'); plt.xlabel('Predicted'); plt.title('Confusion Matrix')
        plt.show()

    
    # hyperparameter tuning

    def hyperparameter_tuning(train_dataset, val_dataset, classes,
                              lrs=[1e-3, 1e-4], batch_sizes=[32, 64], tune_epochs=4):
        best_score = -1
        best_cfg = None
        results = []
        for lr in lrs:
            for bs in batch_sizes:
                print(f"\nTuning: lr={lr}, batch_size={bs}")
                train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS)
                val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)
                model = AdvancedCNN(num_classes=len(classes)).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=lr)
                # quick training
                for epoch in range(tune_epochs):
                    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
                    val_loss, val_acc, prec, recall, f1, _, _ = validate(model, val_loader, criterion)
                    print(f"  Ep {epoch+1}/{tune_epochs} | train_acc {train_acc:.3f} | val_acc {val_acc:.3f} | val_f1 {f1:.3f}")
                # evaluate after tune_epochs
                if f1 > best_score:
                    best_score = f1
                    best_cfg = {'lr': lr, 'batch_size': bs}
                results.append({'lr': lr, 'batch_size': bs, 'val_f1': f1})
        print("\nTuning results:", results)
        print("Best cfg:", best_cfg, "best val_f1:", best_score)
        return best_cfg
    
     #full training

    def train_full(train_dataset, val_dataset, test_dataset, classes,
                      lr=1e-3, batch_size=32, num_epochs=12, save_path='best_model.pth'):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)



