#––––––––––––––––––––––––––––––––––––––
#             FASTER VERSION
#______________________________________



# this is the same version of the code found in Samuele.Magatti.ML.Project 
# but it was chosen to speed up the code in these ways:
# reduce the number of epochs from 10 to 5 ( or from 3 to 2 in the hyperparameter tunung)
# reduce the resize size ( from the previous 300x200 to 128x128)
# this shold speed up the code but with a loss in accouracy of 1 or 2 %
# the analisis and the graphs has been done using the previous version of the code 

import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image

# set seed

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# set seet to guarantee reproducibility of data augmentation ( the seed_worker function must be before the if __name__ == "__main__": )

def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

       
g = torch.Generator().manual_seed(SEED)

# Check if you have installed data correctlyi

import os
if __name__ == "__main__": # avoid num_workers=4 to replicate each part of the code

    if not os.path.exists ("data"):
       raise FileNotFoundError("put the data in a folder called 'data/' as written in the README file")

    #---------------------------
    #    DATA EXPLORATION 
    #---------------------------

    ## From the readme I know that:
    # All images are RGB images of 300 pixels wide by 200 pixels high in .png format. 
    # the images are separated in three sub-folders named 'rock', 'paper' and 'scissors' according to their respective class.


    #Check if the data are images and count the total number of images in all the files in data folder

    import glob


    images = glob.glob("data/**/*.png", recursive=True)

    if len(images) == 0:
        raise FileNotFoundError("No images found in data/ or its subfolders (expected .png files)")
    else:
        print(f"Found {len(images)} PNG files.")
    

    # find and eliminate corrupted files

    from PIL import Image
    import os

    base_dir = "data"

    classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    corrupted = []  
    valid_extensions = {".png"}

    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        for f in os.listdir(class_dir):
            img_path = os.path.join(class_dir, f)

            # check if it's an image
            if not os.path.isfile(img_path):
                continue
            if not any(f.lower().endswith(ext) for ext in valid_extensions):
                continue

            try:
                with Image.open(img_path) as img:
                    img.verify() 
            except Exception as e:
                print(f" corruoted image: {img_path} ({e})")
                corrupted.append(img_path)

    print(f"\nTotal corrupted images: {len(corrupted)}")

    # eliminate corrupted images

    for img_path in corrupted:
        os.remove(img_path)  

    # find and eliminate duplicate images

    import os
    import hashlib

    # compute the hash MD5 of a file

    def hash_file(file_path):
    
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    base_dir = "data"
    duplicates = {}
    duplicates_to_remove = []

    for root, dirs, files in os.walk(base_dir):
        if "rps-cv-images" in root:
            continue
        for f in files:
            if f.lower().endswith((".png")):
                path = os.path.join(root, f)
                h = hash_file(path)
                if h in duplicates:
                    print(f"Duplicate found: {path} is identical to {duplicates[h]}")
                    duplicates_to_remove.append(path)
                else:
                    duplicates[h] = path

    print(f"\nTotal duplicates found: {len(duplicates_to_remove)}")

    # eliminate duplicates

    for path in duplicates_to_remove:
        os.remove(path)

    #number of images for each class

    import os


    # Take ino account only the main folder data 

    counts = {
        cls: len([
            f for f in os.listdir(os.path.join(base_dir, cls))
            if f.lower().endswith(('.png'))
        ])
        for cls in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, cls)) and cls != "rps-cv-images"
    }

    print(counts)


    # barplot of the number of images in each class

    import matplotlib.pyplot as plt


    # print the results in the console

    for cls, count in counts.items():
        print(f"{cls}: {count} file")

    # create the plot 

    plt.bar(counts.keys(), counts.values())
    plt.title("Number of files in each class", color="darkblue")
    plt.suptitle("Dataset Rock-Paper-Scissors", fontsize=18, fontweight='bold')
    plt.xlabel("Class", color="darkblue")
    plt.ylabel("Number of images", color="darkblue")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # image size and aspect ratio

    import os
    from PIL import Image
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    #check the size of each image 

    # create a dictionary list with name, width and height
    data = []

    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png')):
                path = os.path.join(class_dir, filename)
                with Image.open(path) as img:
                    width, height = img.size
                    data.append({
                        "filename": filename,
                        "width": width,
                        "height": height,
                        "class": cls
                    })
 
    # Create the DataFrame
 
    df = pd.DataFrame(data)
 
    # Show the first rows
 
    print(df.head())
 
    # plot the distribution of width and height
 
    plt.figure(figsize=(10,5))
    sns.histplot(df["width"], color="skyblue", label="Width", kde=True)
    sns.histplot(df["height"], color="salmon", label="Height", kde=True)
    plt.title("width and height distribution of images")
    plt.xlabel("Dimension (pixel)")
    plt.ylabel("number of images")
    plt.legend()
    plt.show()

    #check the aspect ratio

    # Compute the ratio 
    df["aspect_ratio"] = df["width"] / df["height"]

    # Show the first rows
    print(df.head())

    # plot the histogram
    df["aspect_ratio"].hist(bins=30)
    plt.title(" Aspect ratio distribution)")
    plt.xlabel("Aspect ratio (width / height)")
    plt.ylabel("Number of images")
    plt.show()

    #check all the images are rgb

    total = 0
    non_rgb = 0
    valid_extensions = { ".png" }

    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            if not os.path.isfile(img_path):
                continue
            if not any(img_name.lower().endswith(ext) for ext in valid_extensions):
                continue
            try:
                with Image.open(img_path) as img:
                    total += 1
                    if img.mode != "RGB":
                        non_rgb += 1
                        print(f"{img_path} is in {img.mode}")
            except Exception as e:
                print(f"error in file {img_path}")

    print(f"\n total images controlled: {total}")
    print(f" non RGB images: {non_rgb}")


    # mean for the rgb values for each image 

    import numpy as np
    r_means, g_means, b_means = [], [], []

    for cls in classes:
        class_dir = os.path.join(base_dir, cls)
        for f in os.listdir(class_dir):
            img_path = os.path.join(class_dir, f)
            if not os.path.isfile(img_path):
                continue
            if not any(f.lower().endswith(ext) for ext in valid_extensions):
                continue
            img = np.array(Image.open(os.path.join(base_dir, cls, f))) / 255.0
            r_means.append(img[...,0].mean())
            g_means.append(img[...,1].mean())
            b_means.append(img[...,2].mean())



    #compute the mean and the standard deviatio for the whole dataset 

    r_mean = np.mean(r_means)
    g_mean = np.mean(g_means)
    b_mean = np.mean(b_means)


    r_std = np.std(r_means)
    g_std = np.std(g_means)
    b_std = np.std(b_means)

    print(f"Mean:  R={r_mean:.4f}, G={g_mean:.4f}, B={b_mean:.4f}")
    print(f"Std:   R={r_std:.4f}, G={g_std:.4f}, B={b_std:.4f}")

    #histogram

    plt.hist(r_means, bins=30, alpha=0.5, color='r', label='R')
    plt.hist(g_means, bins=30, alpha=0.5, color='g', label='G')
    plt.hist(b_means, bins=30, alpha=0.5, color='b', label='B')
    plt.axvline(r_mean, color='r', linestyle='--', linewidth=2, label=f'R mean = {r_mean:.2f}')
    plt.axvline(g_mean, color='g', linestyle='--', linewidth=2, label=f'G mean = {g_mean:.2f}')
    plt.axvline(b_mean, color='b', linestyle='--', linewidth=2, label=f'B mean = {b_mean:.2f}')
    plt.legend() 
    plt.title("distribution of average values of rgb")
    plt.xlabel("average value")
    plt.ylabel("frequency")
    plt.legend(loc='upper center', bbox_to_anchor=(1, 1))
    plt.show()


    #-----------------------------------
    #           PREPROCESSING
    #-----------------------------------

    import torch
    import random
    import numpy as np
    from torch.utils.data import DataLoader, random_split
    from torchvision import datasets, transforms
    from PIL import Image
    import sys

    ## image resizing, normalization and data augmentation

    g = torch.Generator().manual_seed(SEED)


    ## split data into train, validation and test set 
    # the images of the data folder and of the data/rps-cv-images are the same 
    #so from now on we will work only on the latter subfolder

    full_dataset = datasets.ImageFolder(root="data/rps-cv-images")

    # Split train test validation (70-15-15)

    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size


    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=(g)
    )

    #compute mean and std for the train part

    r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
    r_sq_sum, g_sq_sum, b_sq_sum = 0.0, 0.0, 0.0
    num_pixels = 0

    for img, _ in train_dataset:
        img = np.array(img) / 255.0  # converte in [0,1]
        if img.ndim == 3 and img.shape[2] == 3:  # controlla che sia RGB
            r = img[:, :, 0]
            g = img[:, :, 1]
            b = img[:, :, 2]
            r_sum += r.mean()
            g_sum += g.mean()
            b_sum += b.mean()
            r_sq_sum += (r ** 2).mean()
            g_sq_sum += (g ** 2).mean()
            b_sq_sum += (b ** 2).mean()
            num_pixels += 1

    train_r_mean = r_sum / num_pixels
    train_g_mean = g_sum / num_pixels
    train_b_mean = b_sum / num_pixels

    train_r_std = (r_sq_sum / num_pixels - train_r_mean ** 2) ** 0.5
    train_g_std = (g_sq_sum / num_pixels - train_g_mean ** 2) ** 0.5
    train_b_std = (b_sq_sum / num_pixels - train_b_mean ** 2) ** 0.5

    print(f"Train Mean:  R={train_r_mean:.4f}, G={train_g_mean:.4f}, B={train_b_mean:.4f}")
    print(f"Train Std:   R={train_r_std:.4f}, G={train_g_std:.4f}, B={train_b_std:.4f}")

    ## From the eda it's known that all images are 200x300, nevertheless here there is a resize check to avoid errors

    ##trasformations for the train set:
    # Resize all the images to 128x128, tensor conversion, RGB normalizarion, data augmentation

    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),     
        transforms.RandomRotation(20),         
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3220,0.5481,0.2593], std=[0.2556,0.1014,0.1329])  # using the previously computed values
    ])

    # trasformation for the validation and test set:
    #resize, tensor conversion and normalization

    val_test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3220,0.5481,0.2593], std=[0.2556,0.1014,0.1329]) # using the train set values
    ])


    # apply the data augmentation and resizing defined before 

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform


    # dataloader setup

    num_workers = 0 if sys.platform == "darwin" else 4 # to avoid errors if you are using a mac device

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=num_workers, worker_init_fn=seed_worker )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                             num_workers=num_workers, worker_init_fn=seed_worker )

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=num_workers, worker_init_fn=seed_worker )


    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")


 
    #-----------------------------------------------
    #                FIRST CNN 
    #-----------------------------------------------


    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # SIMPLE CNN MODEL 

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
    

    
    # TRAINING PARAMETERS

    num_epochs = 5
    lr = 1e-3
    batch_size = 32
    num_classes = 3

    # DATALOADER

    model = TinyCNN(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # STORAGE METRICHE

    train_losses,val_losses, val_accuracies, train_accuracies = [], [], [], []

        # TRAINING LOOP ( 10 epochs)


    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0


    
        # mini-batch training
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs, labels)  # compute the loss
            loss.backward()   # backpropagation
            optimizer.step()  # update weights
        
            #  compute average loss and accuracy of the batch
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        # compute averages for each epoch
        epoch_loss = running_loss / total
        epoch_acc = correct / total
    
        # VALIDATION

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0


        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        # save the data for the plot
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}" )


        
    # GRAPH

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # FINAL TEST

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = test_correct / test_total
    print(f"\n Test Accuracy: {test_accuracy:.4f}")

    # CONFUSION MATRIX

    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            out = model(x.to(device))
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
   
# MODEL EVALUATION

    model.eval()
    y_true, y_pred = [], []


    # prediction on validation set

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    class_names = ['Rock', 'Paper', 'Scissors']

    # detailed report

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    df_report = pd.DataFrame(report).transpose()

    # plot

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    ## LABEL SHUFFLE TEST

    import copy, random

    
    # the test accuracy seem too high: train a label shuffle

    # train dataset copy

    shuffled_train = copy.deepcopy(train_dataset)
    targets = [y for _, y in shuffled_train]
    random.shuffle(targets)

    # substitute the labels

    for i, idx in enumerate(shuffled_train.indices):
        path, _ = shuffled_train.dataset.samples[idx]
        shuffled_train.dataset.samples[idx] = (path, targets[i])

    # new dataloader

    shuffled_loader = DataLoader(shuffled_train, batch_size=32, shuffle=True,
                             num_workers=num_workers, worker_init_fn=seed_worker)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_shuffle = TinyCNN(num_classes=3).to(device)


    #optimization and loss (crossentropy loss and learning rate=0.001)


    optimizer = optim.Adam(model_shuffle.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 2 epoch train

    num_epochs = 1
    for epoch in range(num_epochs):
        model_shuffle.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in shuffled_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_shuffle(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/total:.4f} - Accuracy: {acc:.2f}%")

    print("Test label shuffle completed.")

    # the data seem to be around 33% so there is no data leakadge
        
    #confusion matrix

    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np


    class_names = ['Rock', 'Paper', 'Scissors']


    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)

    df_report = pd.DataFrame(report).transpose()

    y_true, y_pred = [], []
    model_shuffle.eval()
    with torch.no_grad():
        for x,y in test_loader:
            out = model_shuffle(x.to(device))
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

    # plot

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
   
    

    

    

    #-----------------------------------------------
    #                  SECOOND CNN
    #-----------------------------------------------


    import seaborn as sns
    import numpy as np
    import pandas as pd
    from sklearn.metrics import classification_report, confusion_matrix
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import matplotlib.pyplot as plt

    #MODEL
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

    #  SETUP TRAINING

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=3).to(device)
    print(model)

    #optimization and loss (crossentropy loss and learning rate=0.001)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # storage for the graphs

    train_losses,val_losses, val_accuracies, train_accuracies = [], [], [], []



    # TRAINING LOOP ( 10 epochs)

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0


    
        # mini-batch training
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
    
            optimizer.zero_grad()
            outputs = model(inputs) 
            loss = criterion(outputs, labels)  # compute the loss
            loss.backward()   # backpropagation
            optimizer.step()  # update weights
        
            #  compute average loss and accuracy of the batch
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
        # compute averages for each epoch
        epoch_loss = running_loss / total
        epoch_acc = correct / total
    
        # VALIDATION

        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0


        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        # save the data for the plot
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
    
        print(f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
            f"Val loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}" )


        
    # GRAPH

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # FINAL TEST

    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = test_correct / test_total
    print(f"\n Test Accuracy: {test_accuracy:.4f}")

    # CONFUSION MATRIX

    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np
    y_true, y_pred = [], []
    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
            out = model(x.to(device))
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))
   

   # MODEL EVALUATION

    model.eval()
    y_true, y_pred = [], []


    # prediction on validation set

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    class_names = ['Rock', 'Paper', 'Scissors']

    # detailed report

    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    df_report = pd.DataFrame(report).transpose()

    # plot

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    
    ## LABEL SHUFFLE TEST

    import copy, random

    
    # the test accuracy seem too high: train a label shuffle

    # train dataset copy

    shuffled_train = copy.deepcopy(train_dataset)
    targets = [y for _, y in shuffled_train]
    random.shuffle(targets)

    # substitute the labels

    for i, idx in enumerate(shuffled_train.indices):
        path, _ = shuffled_train.dataset.samples[idx]
        shuffled_train.dataset.samples[idx] = (path, targets[i])

    # new dataloader

    shuffled_loader = DataLoader(shuffled_train, batch_size=32, shuffle=True,
                             num_workers=num_workers, worker_init_fn=seed_worker)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_shuffle = SimpleCNN(num_classes=3).to(device)


    #optimization and loss (crossentropy loss and learning rate=0.001)


    optimizer = optim.Adam(model_shuffle.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 2 epoch train

    num_epochs = 1
    for epoch in range(num_epochs):
        model_shuffle.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in shuffled_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model_shuffle(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/total:.4f} - Accuracy: {acc:.2f}%")

    print("Test label shuffle completed.")

    # the data seem to be around 33% so there is no data leakadge
        
    #confusion matrix

    from sklearn.metrics import confusion_matrix, classification_report
    import numpy as np


    class_names = ['Rock', 'Paper', 'Scissors']


    report = classification_report(y_true, y_pred, target_names=class_names, 
                                   output_dict=True, zero_division=0)

    df_report = pd.DataFrame(report).transpose()

    y_true, y_pred = [], []
    model_shuffle.eval()
    with torch.no_grad():
        for x,y in test_loader:
            out = model_shuffle(x.to(device))
            y_pred.extend(out.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())

    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, digits=4))

    # plot

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
   


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

        model = AdvancedCNN(num_classes=len(classes)).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        best_model_wts = copy.deepcopy(model.state_dict())
        best_f1 = -1

        history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[], 'val_f1':[]}

        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
            val_loss, val_acc, prec, recall, f1, y_true_val, y_pred_val = validate(model, val_loader, criterion)
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(f1)

            print(f"Epoch {epoch+1}/{num_epochs} | Tr Loss {train_loss:.4f} Acc {train_acc:.4f} | Val Loss {val_loss:.4f} Acc {val_acc:.4f} F1 {f1:.4f} | time {(time.time()-t0):.1f}s")

            # save best
            if f1 > best_f1:
                best_f1 = f1
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({'model_state': best_model_wts, 'classes': classes}, save_path)

        # load best weights
        model.load_state_dict(best_model_wts)

        # final test evaluation
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        model.eval()
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_trues.extend(labels.cpu().numpy())
        acc = accuracy_score(all_trues, all_preds)
        prec, recall, f1, _ = precision_recall_fscore_support(all_trues, all_preds, average='weighted', zero_division=0)
        print(f"\nTest results -> Acc: {acc:.4f} Prec: {prec:.4f} Recall: {recall:.4f} F1: {f1:.4f}")
        return model, history, (all_trues, all_preds), test_loader

    #USAGE

    classes = ['rock','paper','scissors'] 

    #  hyperparameter tuning (short runs)
    best_cfg = hyperparameter_tuning(train_dataset, val_dataset, classes,
                                     lrs=[1e-3, 3e-4, 1e-4], batch_sizes=[32, 64], tune_epochs=2)

    #  full train with best config (longer)
    chosen_lr = best_cfg['lr'] if best_cfg else 1e-3
    chosen_bs = best_cfg['batch_size'] if best_cfg else 32
    model, history, (y_test, y_pred), test_loader = train_full(train_dataset, val_dataset, test_dataset, classes,
                                                               lr=chosen_lr, batch_size=chosen_bs, num_epochs=5,
                                                               save_path='best_advanced_cnn.pth')

    # plots and metrics
    plot_training(history)
    print("\nClassification report (test):")
    print(classification_report(y_test, y_pred, target_names=classes, digits=4))
    plot_confusion(y_test, y_pred, classes)


    #  discussion helpers: overfitting/underfitting

    train_final_acc = history['train_acc'][-1]
    val_final_acc = history['val_acc'][-1]
    print(f"\nTrain final acc: {train_final_acc:.4f}, Val final acc: {val_final_acc:.4f}")
    if train_final_acc - val_final_acc > 0.08:
        print("Possible overfitting detected (train >> val). Consider stronger regularization or more augmentation.")
    elif val_final_acc - train_final_acc > 0.05:
        print("Possible underfitting (val > train) or noisy training — check learning rate / model capacity.")
    else:
        print("No strong over/underfitting signal from final accuracies.")

   

