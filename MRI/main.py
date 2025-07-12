import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import shutil
import os
from MRIDataset import MRIDataset
from CustomAlexNet import CustomAlexNet

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load your dataset
datapath = 'E:\\pakinsonreviewcasestudy\\MRI\\data\\'
dataset = MRIDataset(datapath, transform=transform)


# Function to calculate metrics
def calculate_metrics(preds, labels):
    tp = torch.sum((preds == 1) & (labels == 1)).item()
    tn = torch.sum((preds == 0) & (labels == 0)).item()
    fp = torch.sum((preds == 1) & (labels == 0)).item()
    fn = torch.sum((preds == 0) & (labels == 1)).item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(labels, preds)
    auc = roc_auc_score(labels, preds)

    return accuracy, recall, specificity, precision, f1, auc


# Training the model
def train_model(model, criterion, optimizer, dataloader, num_epochs=25):
    epoch_loss_list = []
    epoch_acc_list = []
    epoch_recall_list = []
    epoch_specificity_list = []
    epoch_precision_list = []
    epoch_f1_list = []
    epoch_auc_list = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in tqdm(dataloader, desc=f"Training Epoch {epoch}"):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)

        epoch_loss = running_loss / len(dataloader.dataset)
        epoch_acc, epoch_recall, epoch_specificity, epoch_precision, epoch_f1, epoch_auc = calculate_metrics(all_preds, all_labels)

        print(
            f'Epoch {epoch} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Recall: {epoch_recall:.4f} Specificity: {epoch_specificity:.4f} Precision: {epoch_precision:.4f} F1: {epoch_f1:.4f} AUC: {epoch_auc:.4f}')
        print("Logging to TensorBoard...")

        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Recall/train', epoch_recall, epoch)
        writer.add_scalar('Specificity/train', epoch_specificity, epoch)
        writer.add_scalar('Precision/train', epoch_precision, epoch)
        writer.add_scalar('F1/train', epoch_f1, epoch)
        writer.add_scalar('AUC/train', epoch_auc, epoch)
        print("Logged to TensorBoard")

        epoch_loss_list.append(epoch_loss)
        epoch_acc_list.append(epoch_acc)
        epoch_recall_list.append(epoch_recall)
        epoch_specificity_list.append(epoch_specificity)
        epoch_precision_list.append(epoch_precision)
        epoch_f1_list.append(epoch_f1)
        epoch_auc_list.append(epoch_auc)

    writer.flush()
    metrics_df = pd.DataFrame({
        'Epoch': range(num_epochs),
        'Loss': epoch_loss_list,
        'Accuracy': epoch_acc_list,
        'Recall': epoch_recall_list,
        'Specificity': epoch_specificity_list,
        'Precision': epoch_precision_list,
        'F1': epoch_f1_list,
        'AUC': epoch_auc_list
    })
    metrics_df.to_csv('training_metrics.csv', index=False)
    return model


# Evaluation
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy, recall, specificity, precision, f1, auc = calculate_metrics(all_preds, all_labels)
    return accuracy, recall, specificity, precision, f1, auc


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_indices, val_indices) in enumerate(kf.split(dataset)):
        print(f'Fold {fold + 1}')

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

        # Load the pre-trained AlexNet model
        alexnet = CustomAlexNet()
        alexnet = alexnet.to(device)

        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(alexnet.parameters(), lr=0.001)

        # Clean and create log directory
        log_dir = f'logs/mri_classification/fold_{fold + 1}'
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir)

        # TensorBoard writer
        writer = SummaryWriter(log_dir)

        # Train the model
        trained_model = train_model(alexnet, criterion, optimizer, train_loader, num_epochs=25)

        # Save the trained model
        torch.save(trained_model.state_dict(), f'alexnet_mri_classification_fold_{fold + 1}.pth')

        # Evaluate the model
        val_accuracy, val_recall, val_specificity, val_precision, val_f1, val_auc = evaluate_model(trained_model, val_loader)
        print(
            f'Validation Accuracy: {val_accuracy:.4f} Recall: {val_recall:.4f} Specificity: {val_specificity:.4f} Precision: {val_precision:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}')

        # Log validation metrics to TensorBoard
        writer.add_scalar('Accuracy/val', val_accuracy, 0)
        writer.add_scalar('Recall/val', val_recall, 0)
        writer.add_scalar('Specificity/val', val_specificity, 0)
        writer.add_scalar('Precision/val', val_precision, 0)
        writer.add_scalar('F1/val', val_f1, 0)
        writer.add_scalar('AUC/val', val_auc, 0)

        # Close the TensorBoard writer
        writer.close()

        fold_results.append({
            'Fold': fold + 1,
            'Accuracy': val_accuracy,
            'Recall': val_recall,
            'Specificity': val_specificity,
            'Precision': val_precision,
            'F1': val_f1,
            'AUC': val_auc
        })

    # Save fold results
    fold_results_df = pd.DataFrame(fold_results)
    fold_results_df.to_csv('kfold_results.csv', index=False)
    print(fold_results_df)
