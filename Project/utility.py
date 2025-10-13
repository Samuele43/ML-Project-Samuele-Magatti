# UTILITY

import torch
import random
import numpy as np


# set seed

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# set seet to guarantee reproducibility of data augmentation ( the seed_worker function must be before the if __name__ == "__main__": )

def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


# Check if you have installed data correctlyi

import os

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
