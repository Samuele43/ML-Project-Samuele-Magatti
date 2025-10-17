# FIRST CNN

from data import get_train_val_test_loaders
train_loader, val_loader, test_loader = get_train_val_test_loaders()
from model import TinyCNN
import torch


# TRAINING PARAMETERS
if __name__ == "__main__" :

    num_epochs = 10
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

    num_epochs = 2
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