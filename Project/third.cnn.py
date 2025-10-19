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
RESIZE = (150, 225)                # H, W (keeps 2:3 aspect ratio)
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
