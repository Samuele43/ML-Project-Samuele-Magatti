
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from PIL import Image


def get_train_val_test_loaders(data_dir="data/rps-cv-images", batch_size=32, seed=42):
    from utils import set_seed, seed_worker

    set_seed(42)


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
        transforms.Resize((150, 225)),
        transforms.RandomHorizontalFlip(),     
        transforms.RandomRotation(15),         
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3220,0.5481,0.2593], std=[0.2556,0.1014,0.1329])  # using the previously computed values
    ])

    # trasformation for the validation and test set:
    #resize, tensor conversion and normalization

    val_test_transform = transforms.Compose([
        transforms.Resize((150, 225)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.3220,0.5481,0.2593], std=[0.2556,0.1014,0.1329]) # using the train set values
    ])


    # apply the data augmentation and resizing defined before 

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform


    # dataloader setup

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              num_workers=4, worker_init_fn=seed_worker, generator=g)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                             num_workers=4, worker_init_fn=seed_worker, generator=g)

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                             num_workers=4, worker_init_fn=seed_worker, generator=g)


    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader
