import os
import torch #DL
import pandas as pd #saving logs
import matplotlib.pyplot as plt #plotting
from torchvision.models import ResNet34_Weights
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay # test evaluation metrics

# 1. Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Dataset paths
base_dir = r'C:/datasets/crops_balanced_split'
train_dir = os.path.join(base_dir, "train")
val_dir   = os.path.join(base_dir, "val")
test_dir  = os.path.join(base_dir, "test")

# 3. Transforms - augmentation
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# 4. Load each split
#loads images anf auto assigns labels
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(root=val_dir,   transform=val_transform)
test_dataset  = datasets.ImageFolder(root=test_dir,  transform=val_transform)

#wrap datasets -training and evaluation
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=16)
test_loader  = DataLoader(test_dataset, batch_size=16)

# 5. Load ResNet34 (no warnings)
weights = ResNet34_Weights.DEFAULT
model = models.resnet34(weights=weights) #speeding up convergence
num_classes = len(train_dataset.classes)
model.fc = nn.Linear(model.fc.in_features, num_classes) #final fully connected layer to match your number of classes
model = model.to(device) #moves to GPU

# 6. Loss & optimizer
criterion = nn.CrossEntropyLoss() #for multi calss classification
optimizer = optim.Adam(model.parameters(), lr=0.0005) #for adjusting weights; learning rate was 0.001

# 7. Training loop
history = {'epoch': [], 'train_loss': [], 'val_accuracy': []}
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad() #gradients
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_acc = 100 * correct / total
    #storesepoch metrics
    history['epoch'].append(epoch + 1)
    history['train_loss'].append(total_loss)
    history['val_accuracy'].append(val_acc)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Val Acc: {val_acc:.2f}%")

# Save model
torch.save(model.state_dict(), 'resnet34_v63_defect_classifier.pth')
print("Model saved as resnet34_v63_defect_classifier.pth")

# Save training log
df = pd.DataFrame(history)
df.to_csv('resnet34_v63_training_results.csv', index=False)
print("Saved training log to resnet34_v63_training_results.csv")

# Plot loss & accuracy
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.legend()
plt.subplot(1,2,2)
plt.plot(history['epoch'], history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy (%)'); plt.title('Validation Accuracy'); plt.legend()
plt.tight_layout()
plt.savefig('resnet34_v63_training_plot.png')
print("Saved plot as resnet34_v63_training_plot.png")

# 8. Evaluate on test set & SAVE REPORTS
model.eval()
test_preds, test_labels = [], []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_preds.extend(predicted.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())
test_acc = 100 * sum([p == l for p, l in zip(test_preds, test_labels)]) / len(test_labels)
print(f"Test Accuracy: {test_acc:.2f}%")


# Generates accuracy, precision, recall, F1-score/class
# Save classification report
test_report = classification_report(test_labels, test_preds, target_names=train_dataset.classes, output_dict=True)
test_report_df = pd.DataFrame(test_report).transpose()
test_report_df.to_csv('resnet34_v63_test_classification_report.csv')
print("Saved test classification report as resnet34_v63_test_classification_report.csv")

# Save confusion matrix plot
cm = confusion_matrix(test_labels, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=train_dataset.classes)
disp.plot()
plt.title('ResNet34_v63 Test Confusion Matrix')
plt.savefig("resnet34_v63_test_confusion_matrix.png")
print("Saved test confusion matrix as resnet34_v63_test_confusion_matrix.png")
