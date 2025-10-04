#ML project

# Check if you have installed data correctly
import os

if not os.path.exists ("data")
   raise FileNotFoundError("put the data in a folder called 'data/' as written in the README file")

#Check if the data are images

import glob

images = glob.glob("data/*.jpg") + glob.glob("data/*.png")
if len(images) == 0:
    raise FileNotFoundError("No images found in data/ (expected .jpg or .png files)")



#Data exploration 




# Preprocessing



## image resizing

## normalization

## data augmentation

## split data into train and test set




required_files = ["data/train.csv", "data/test.csv"]

for f in required_files:
    if not os.path.exists(f):
        raise FileNotFoundError(f"Missing {f}. Put the data in a folder called 'data/' as written in the README file")



#build of first CNN













