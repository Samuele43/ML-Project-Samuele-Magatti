#--------------------------------
#    MACHINE LEARNING PROJECT
#--------------------------------

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

    ## image resizing, normalization and data augmentation

    g = torch.Generator()
    g.manual_seed(SEED)


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
        generator=torch.Generator().manual_seed(SEED)
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
    # Resize all the images to 200x300, tensor conversion, RGB normalizarion, data augmentation

    train_transform = transforms.Compose([
        transforms.Resize((200, 300)),
        transforms.RandomHorizontalFlip(),     
        transforms.RandomRotation(20),         
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3220,0.5481,0.2593], std=[0.2556,0.1014,0.1329])  # using the previously computed values
    ])

    # trasformation for the validation and test set:
    #resize, tensor conversion and normalization

    val_test_transform = transforms.Compose([
        transforms.Resize((200, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3220,0.5481,0.2593], std=[0.2556,0.1014,0.1329]) # using the train set values
    ])


    # apply the data augmentation and resizing defined before 

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform


    # dataloader setup

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=4, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED))

    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                             num_workers=4, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED))

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=4, worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED))


    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")




    #-----------------------------------------------
    #                  FIRST CNN
    #-----------------------------------------------



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

    num_epochs = 10

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
    
    # the test accuracy seem too high: train a label reshuffle

    
    #confusion matrix

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
   
   


#-----------------------------------------------
#                  SECOND CNN
#-----------------------------------------------



