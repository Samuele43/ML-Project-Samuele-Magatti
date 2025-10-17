# SECOND CNN 
from data import get_train_val_test_loaders
train_loader, val_loader, test_loader = get_train_val_test_loaders()
from model import SimpleCNN
import torch

#  SETUP TRAINING
if __name__ == "__main__":
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

    #confusion matrix

    from utility import plot_confusion_matrix


    class_names = full_dataset.classes  # recupera i nomi delle classi ('rock', 'paper', 'scissors')
    plot_confusion_matrix(model, test_loader, device, class_names, title="Confusion Matrix - SimpleCNN")
    plot_confusion_matrix(model_shuffle, test_loader, device, class_names, title="Confusion Matrix - Label Shuffle Test")


   