#ML project

# Check if you have installed data correctly

import os

if not os.path.exists ("data"):
   raise FileNotFoundError("put the data in a folder called 'data/' as written in the README file")

#Check if the data are images and count the total number of images in all the files in data folder

import glob

images = glob.glob("data/**/*.png", recursive=True)

if len(images) == 0:
    raise FileNotFoundError("No images found in data/ or its subfolders (expected .png files)")
else:
    print(f"Found {len(images)} PNG files.")
    

#Data exploration 
## From the readme I know that:
# All images are RGB images of 300 pixels wide by 200 pixels high in .png format. 
# the images are separated in three sub-folders named 'rock', 'paper' and 'scissors' according to their respective class.

#number of images for each class

import os

base_dir = "data"

# Take ino account only the main folder data 

counts = {
    cls: len([
        f for f in os.listdir(os.path.join(base_dir, cls))
        if f.lower().endswith(('.jpg', '.png'))
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

base_dir = "data" 

#list of classes

classes = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]


# create a dictionary list with name, width and height
data = []

for cls in classes:
    class_dir = os.path.join(base_dir, cls)
    for filename in os.listdir(class_dir):
        if filename.lower().endswith(('.jpg', '.png')):
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


#resize (?)


# mean for the rgb values

import numpy as np
r_means, g_means, b_means = [], [], []

for cls in classes:
    for f in os.listdir(os.path.join(base_dir, cls))[:20]:
        img = np.array(Image.open(os.path.join(base_dir, cls, f))) / 255.0
        r_means.append(img[...,0].mean())
        g_means.append(img[...,1].mean())
        b_means.append(img[...,2].mean())

plt.hist(r_means, bins=30, alpha=0.5, label='R')
plt.hist(g_means, bins=30, alpha=0.5, label='G')
plt.hist(b_means, bins=30, alpha=0.5, label='B')
plt.legend(); plt.show()


# Preprocessing



## image resizing

## normalization

# (?) calcola prima i valori della media e della deviazione standard
transforms.Normalize(mean=[?, ?, ?],
                     std=[?, ?, ?])


## data augmentation (?)


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

from sklearn.model_selection import train_test_split

train_paths, test_paths = train_test_split(image_paths, test_size=0.2, stratify=labels, random_state=42)




required_files = ["data/train.csv", "data/test.csv"]

for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing {f}. Put the data in a folder called 'data/' as written in the README file")



#build of first CNN













