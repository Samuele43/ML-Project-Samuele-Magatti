#––––––––––––––––––––––––––––––––––––––
#             FASTER VERSION
#______________________________________



# this is the same version of the code found in Samuele.Magatti.ML.Project 
# but it was chosen to speed up the code in these ways:
# reduce the number of epochs from 10 to 5 
# reduce the resize size ( from the previous 225x150 to 128x128)
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
        transforms.RandomRotation(15),         
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
            # Input: 128x128 -> after 2 pool(2x2): 32x32
            self.fc1 = nn.Linear(32 * 32 * 32, 64)
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
            #  input images: 128x128 -> after 3 pool(2x2): 16x16 (approx)
            self.fc1 = nn.Linear(128 * 16 * 16, 256)
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



    # TRAINING LOOP ( 5 epochs)

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

    import os
    import sys
    import time
    import copy
    import random
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
    from torchvision import datasets, transforms
    from PIL import Image
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm import tqdm

# CONFIGURATION AND SEED
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0 if sys.platform == "darwin" else 4

DATA_PATH = "data/rps-cv-images"   # ImageFolder root
EXTERNAL_PATH = "check.data"       # external dataset folder (optional)
RESIZE = (128, 128)                # H, W (keeps 2:3 aspect ratio)
TRAINVAL_RATIO = 0.8
K_FOLDS = 3

# helper: seed worker for dataloader
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# compute mean/std from list of indices (opens images with PIL)
def compute_mean_std_from_indices(imagefolder_dataset, indices, resize=None):
    """
    imagefolder_dataset: torchvision.datasets.ImageFolder (no transform)
    indices: list of integer indices into imagefolder_dataset.samples
    resize: (H,W) tuple or None
    returns: mean, std lists (3,)
    """
    
    r_sum = g_sum = b_sum = 0.0
    r_sq = g_sq = b_sq = 0.0
    n = 0

    for idx in indices:
        path, _ = imagefolder_dataset.samples[idx]
        img = Image.open(path).convert("RGB")
        if resize is not None:
            img = img.resize((resize[1], resize[0]))  # PIL uses (W,H)
        arr = np.array(img).astype(np.float32) / 255.0
        r = arr[:, :, 0]; g = arr[:, :, 1]; b = arr[:, :, 2]
        r_sum += r.mean(); g_sum += g.mean(); b_sum += b.mean()
        r_sq += (r**2).mean(); g_sq += (g**2).mean(); b_sq += (b**2).mean()
        n += 1

    mean = np.array([r_sum, g_sum, b_sum]) / n
    std = np.sqrt(np.array([r_sq, g_sq, b_sq]) / n - mean**2)
    return mean.tolist(), std.tolist()

# Subset wrapper to apply transforms per-subset
class SubsetWithTransform(torch.utils.data.Dataset):
    def __init__(self, imagefolder_dataset, indices, transform=None):
        """
        imagefolder_dataset: torchvision.datasets.ImageFolder (no transform)
        indices: list of int indices into imagefolder_dataset.samples
        transform: torchvision transform applied to PIL image
        """
        self.dataset = imagefolder_dataset
        self.indices = list(indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        path, label = self.dataset.samples[real_idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# PREPROCESSING: split 80/20 and create K folds (folds are lists of indices)

def prepare_splits_and_folds(data_path=DATA_PATH, trainval_ratio=TRAINVAL_RATIO, k=K_FOLDS, resize=RESIZE):
    # load ImageFolder WITHOUT transforms (we will open images manually or apply transforms in SubsetWithTransform)
    full_dataset = datasets.ImageFolder(data_path, transform=None)
    total = len(full_dataset)
    trainval_size = int(trainval_ratio * total)
    test_size = total - trainval_size

    rng = np.random.RandomState(SEED)
    all_indices = np.arange(total)
    rng.shuffle(all_indices)
    trainval_indices = all_indices[:trainval_size].tolist()
    test_indices = all_indices[trainval_size:].tolist()

    # create k folds inside trainval (folds are lists of indices into full_dataset)
    fold_size = trainval_size // k
    folds = []
    for i in range(k):
        start = i * fold_size
        end = (i + 1) * fold_size if i < k - 1 else trainval_size
        fold_idx = trainval_indices[start:end]
        folds.append(list(fold_idx))

    # compute mean/std for each fold's training set (i.e., concat of other folds)
    means_stds = []
    for i in range(k):
        train_idx = []
        for j in range(k):
            if j != i:
                train_idx += folds[j]
        mean, std = compute_mean_std_from_indices(full_dataset, train_idx, resize=resize)
        means_stds.append({'mean': mean, 'std': std})
        print(f"[prepare] Fold {i+1}: mean={mean}, std={std}")

    # create transforms per fold (train/val). We'll keep these here for convenience.
    transforms_per_fold = []
    for i in range(k):
        m = means_stds[i]['mean']; s = means_stds[i]['std']
        train_tf = transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=m, std=s)
        ])
        val_tf = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=m, std=s)
        ])
        transforms_per_fold.append({'train_tf': train_tf, 'val_tf': val_tf})

    # also prepare a simple test transform placeholder (will recompute mean/std on TrainVal later)
    test_transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor()
    ])

    print("Preprocessing done.")
    return {
        'full_dataset': full_dataset,
        'trainval_indices': trainval_indices,
        'test_indices': test_indices,
        'folds': folds,
        'transforms_per_fold': transforms_per_fold,
        'means_stds': means_stds,
        'test_transform': test_transform
    }


# MODEL DEFINITIONS

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, pool=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, padding=kernel//2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2,2) if pool else nn.Identity()
    def forward(self, x):
        x = self.conv(x); x = self.bn(x); x = self.relu(x); x = self.pool(x)
        return x

class AdvancedCNN(nn.Module):
    def __init__(self, num_classes=3, dropout_prob=0.4):
        super().__init__()
        self.block1 = nn.Sequential(ConvBlock(3,32), ConvBlock(32,32,pool=False))
        self.block2 = nn.Sequential(ConvBlock(32,64), ConvBlock(64,64,pool=False))
        self.block3 = nn.Sequential(ConvBlock(64,128), ConvBlock(128,128,pool=False))
        self.block4 = nn.Sequential(ConvBlock(128,256), ConvBlock(256,256,pool=False))
        self.global_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x); x = self.block4(x)
        x = self.global_pool(x); x = x.view(x.size(0), -1); x = self.dropout(x)
        return self.fc(x)

# TRAIN  AND EVALUATION HELPERS

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0; correct = 0; total = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss/total, correct/total

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0; correct = 0; total = 0
    ys, ps = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            ys.extend(labels.cpu().numpy()); ps.extend(preds.cpu().numpy())
    acc = correct/total
    prec, recall, f1, _ = precision_recall_fscore_support(ys, ps, average='weighted', zero_division=0)
    return running_loss/total, acc, prec, recall, f1, np.array(ys), np.array(ps)


# GRID SEARCH CV (uses folds as lists of indices)

def grid_search_cv(full_dataset, folds, classes, param_grid, k=K_FOLDS, epochs=2, resize=RESIZE):
    """
    full_dataset: ImageFolder (no transforms)
    folds: list of k lists of indices (into full_dataset)
    """
    best_f1 = -1.0
    best_params = None

    for lr in param_grid['lr']:
        for bs in param_grid['batch_size']:
            for dp in param_grid['dropout']:
                print(f"\nGrid Search: lr={lr}, batch_size={bs}, dropout={dp}")
                fold_f1s = []
                for i in range(k):
                    val_idx = folds[i]
                    train_idx = []
                    for j in range(k):
                        if j != i:
                            train_idx += folds[j]

                    # compute mean/std on train_idx
                    mean, std = compute_mean_std_from_indices(full_dataset, train_idx, resize=resize)

                    # transforms
                    train_tf = transforms.Compose([
                        transforms.Resize(resize),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(15),
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
                    ])
                    val_tf = transforms.Compose([
                        transforms.Resize(resize),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=mean, std=std)
                    ])

                    # create datasets and loaders
                    train_ds = SubsetWithTransform(full_dataset, train_idx, transform=train_tf)
                    val_ds = SubsetWithTransform(full_dataset, val_idx, transform=val_tf)

                    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
                    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)

                    # model init
                    model = AdvancedCNN(num_classes=len(classes), dropout_prob=dp).to(device)
                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)

                    # quick training per fold with early stopping (patience = 1)
                    best_val_loss = float('inf'); patience = 1; counter = 0
                    for ep in range(epochs):
                        train_one_epoch(model, train_loader, criterion, optimizer)
                        val_loss, val_acc, _, _, val_f1, _, _ = evaluate(model, val_loader, criterion)
                        if val_loss < best_val_loss - 1e-9:
                            best_val_loss = val_loss; counter = 0
                        else:
                            counter += 1
                        if counter > patience:
                            break

                    _, acc, prec, rec, f1, _, _ = evaluate(model, val_loader, criterion)
                    fold_f1s.append(f1)
                    print(f"  Fold {i+1}/{k} - F1: {f1:.3f} (acc {acc:.3f})")

                avg_f1 = float(np.mean(fold_f1s))
                print(f"Mean F1 over {k} folds: {avg_f1:.3f}")
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_params = {'lr': lr, 'batch_size': bs, 'dropout': dp}
    print("\nBest parameters (by CV mean F1):", best_params, f"(F1={best_f1:.3f})")
    return best_params

# FINAL TRAINING ON TRAINVAL AND EVAL ON TEST 

def final_training_after_tuning(full_dataset, trainval_indices, test_indices, classes, best_params, num_epochs=8, resize=RESIZE, save_path='best_model_trainval.pth'):
    # recompute mean/std on whole TrainVal
    mean, std = compute_mean_std_from_indices(full_dataset, trainval_indices, resize=resize)
    print(f"TrainVal mean/std used: {mean} / {std}")

    trainval_tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    test_tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    trainval_ds = SubsetWithTransform(full_dataset, trainval_indices, transform=trainval_tf)
    test_ds = SubsetWithTransform(full_dataset, test_indices, transform=test_tf)

    bs = best_params['batch_size']; lr = best_params['lr']; dp = best_params['dropout']

    train_loader = DataLoader(trainval_ds, batch_size=bs, shuffle=True, num_workers=NUM_WORKERS, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=NUM_WORKERS)

    model = AdvancedCNN(num_classes=len(classes), dropout_prob=dp).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_wts = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    history = {'train_loss': [], 'train_acc': []}

    for epoch in range(num_epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        _, monitor_acc, _, _, monitor_f1, _, _ = evaluate(model, test_loader, criterion)
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
        print(f"Epoch {epoch+1}/{num_epochs} | Train acc {train_acc:.3f} | Monitor acc {monitor_acc:.3f} | F1 {monitor_f1:.3f} | time {(time.time()-t0):.1f}s")
        if monitor_f1 > best_f1:
            best_f1 = monitor_f1
            best_wts = copy.deepcopy(model.state_dict())
            torch.save({
               'model_state': best_wts,
                'classes': classes,
                'mean': mean,
                'std': std,
                'history': history
            }, save_path)

    model.load_state_dict(best_wts)
    print("Saved best model to", save_path)

    # final evaluation on test
    test_loss, test_acc, prec, recall, f1, y_true, y_pred = evaluate(model, test_loader, criterion)
    print("\nFINAL TEST METRICS")
    print(f"Acc: {test_acc:.4f} | Prec: {prec:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    print("\nClassification report:\n", classification_report(y_true, y_pred, target_names=classes, digits=4))
    return model, test_loader, (y_true, y_pred), history

# GRAPHS

def plot_confusion(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix"); plt.show()

def show_misclassified(model, loader, classes, max_images=12):
    model.eval()
    mis = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            preds = outs.argmax(1)
            mism = preds != labels
            for i in range(imgs.size(0)):
                if mism[i].item():
                    mis.append((imgs[i].cpu(), labels[i].cpu().item(), preds[i].cpu().item()))
                if len(mis) >= max_images:
                    break
            if len(mis) >= max_images:
                break
    if len(mis) == 0:
        print("No misclassified images found.")
        return
    cols = min(6, max_images)
    rows = int(np.ceil(len(mis) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.array(axes).reshape(-1)
    for i, (img, t, p) in enumerate(mis):
        ax = axes[i]
        im = img.permute(1,2,0).numpy()
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
        ax.imshow(im)
        ax.set_title(f"T:{classes[t]} P:{classes[p]}")
        ax.axis("off")
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout(); plt.show()

# EXTERNAL EVALUATION

def evaluate_external(model, external_path, mean, std, resize=RESIZE):
    if not os.path.exists(external_path):
        print("External dataset not found at", external_path); return
    ext_tf = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    ext_ds = datasets.ImageFolder(external_path, transform=ext_tf)
    ext_loader = DataLoader(ext_ds, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for imgs, labels in ext_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outs = model(imgs)
            preds = outs.argmax(1)
            ys.extend(labels.cpu().numpy()); ps.extend(preds.cpu().numpy())
    print("\nExternal dataset classification report:")
    print(classification_report(ys, ps, target_names=ext_ds.classes, digits=4))
    cm = confusion_matrix(ys, ps)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=ext_ds.classes, yticklabels=ext_ds.classes)
    plt.title("External confusion matrix"); plt.show()
    acc = np.mean(np.array(ys) == np.array(ps))
    print(f"External Accuracy: {acc:.4f}")


# MAIN EXECUTION

if __name__ == "__main__":
    # Prepare splits and folds
    data_info = prepare_splits_and_folds(DATA_PATH, TRAINVAL_RATIO, K_FOLDS, RESIZE)
    full_dataset = data_info['full_dataset']
    trainval_indices = data_info['trainval_indices']
    test_indices = data_info['test_indices']
    folds = data_info['folds']
    transforms_per_fold = data_info['transforms_per_fold']
    means_stds = data_info['means_stds']

    classes = full_dataset.classes
    print(f"Total images: {len(full_dataset)} | TrainVal: {len(trainval_indices)} | Test: {len(test_indices)}")
    print("Classes:", classes)

    # small grid (fast)
    param_grid = {'lr':[1e-3, 1e-4], 'batch_size':[32, 64], 'dropout':[0.3, 0.5]}

    # run grid search on trainval folds
    best_params = grid_search_cv(full_dataset, folds, classes, param_grid, k=K_FOLDS, epochs=2, resize=RESIZE)
    if best_params is None:
        raise RuntimeError("Grid search returned no parameters")

    # final training on entire TrainVal and evaluate on Test
    model, test_loader, (y_true, y_pred), history = final_training_after_tuning(
        full_dataset, trainval_indices, test_indices, classes, best_params, num_epochs=8, resize=RESIZE, save_path='best_model_trainval.pth'
    )

    # confusion + misclassified
    plot_confusion(y_true, y_pred, classes)
    show_misclassified(model, test_loader, classes, max_images=8)



    import torch
    import matplotlib.pyplot as plt

    #  LOAD CHECKPOINT 
    ckpt = torch.load('best_model_trainval.pth', map_location='cpu')

    train_loss = ckpt['train_loss']
    train_acc = ckpt['train_acc']

    #  PLOT LOSS 
    plt.figure()
    plt.plot(train_loss, marker='o')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

    #  PLOT ACCURACY 
    plt.figure()
    plt.plot(train_acc, marker='o')
    plt.title('Training Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()


    # external evaluation if checkpoint saved contains mean/std
    ckpt_path = 'best_model_trainval.pth'
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        mean_saved = ckpt.get('mean', None)
        std_saved = ckpt.get('std', None)
        if mean_saved is not None and std_saved is not None:
            evaluate_external(model, EXTERNAL_PATH, mean_saved, std_saved, resize=RESIZE)
        else:
            print("Checkpoint does not contain mean/std; skipping external eval.")
    else:
        print("Checkpoint not found; skipping external eval.")
