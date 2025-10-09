#ML project

# Check if you have installed data correctly

import os

if not os.path.exists ("data"):
   raise FileNotFoundError("put the data in a folder called 'data/' as written in the README file")

#Data exploration 
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
for root, dirs, files in os.walk(base_dir):
    if "rps-cv-images" in root:
        continue
    for f in files:
        if f.lower().endswith((".png")):
            path = os.path.join(root, f)
            h = hash_file(path)
            if h in duplicates:
                print(f"Duplicate found: {path} is identical to {duplicates[h]}")
            else:
                duplicates[h] = path


# eliminate duplicates

for path in duplicates:
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

#check the size of each image for each class

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


# mean for the rgb values

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

#histogram

plt.hist(r_means, bins=30, alpha=0.5, color='r', label='R')
plt.hist(g_means, bins=30, alpha=0.5, color='g', label='G')
plt.hist(b_means, bins=30, alpha=0.5, color='b', label='B')
plt.legend() 
plt.title("distribution of average values of rgb")
plt.xlabel("average value")
plt.ylabel("frequency")
plt.show()

#satndard deviation of rgb values

print("standard deviation R:", np.std(r_means))
print("standard deviation G:", np.std(g_means))
print("standard deviation B:", np.std(b_means))


# Preprocessing

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


## image resizing, normalization and data augmentation

# Resize tutte le immagini a 200x300, conversione in tensor, normalizzazione RGB
train_transform = transforms.Compose([
    transforms.Resize((200, 300)),
    transforms.RandomHorizontalFlip(),     # augmentation: flip casuale
    transforms.RandomRotation(20),         # augmentation: rotazioni casuali ±20°
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

val_transform = transforms.Compose([
    transforms.Resize((200, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])






from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])


## split data into train and test set (?)

#1

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


#2

from sklearn.model_selection import train_test_split

train_paths, test_paths = train_test_split(image_paths, test_size=0.2, stratify=labels, random_state=42)




required_files = ["data/train.csv", "data/test.csv"]

for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing {f}. Put the data in a folder called 'data/' as written in the README file")



#build of first CNN













