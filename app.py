import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter
from utils.featureExtractor import feature_extractor
from utils.signLanguageTransformer import SignLanguageTransformer

# Custom Dataset
class SequenceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.data = []
        self.labels = []

        for seq_name in sorted(os.listdir(root)):
            seq_path = os.path.join(root, seq_name)
            if os.path.isdir(seq_path):
                sequences = {}
                for file_name in sorted(os.listdir(seq_path)):
                    if file_name.endswith('.jpg'):
                        parts = file_name.split('__')
                        seq_num = parts[0]  # Ambil bagian sebelum `__` sebagai urutan
                        if seq_num not in sequences:
                            sequences[seq_num] = []
                        sequences[seq_num].append(os.path.join(seq_path, file_name))
                for seq_files in sequences.values():
                    self.data.append(seq_files)
                    self.labels.append(seq_name)

        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        self.labels = [self.label_to_idx[label] for label in self.labels]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = self.data[idx]
        label = self.labels[idx]
        images = [Image.open(frame).convert('RGB') for frame in frames]
        if self.transform:
            images = [self.transform(img) for img in images]
        return torch.stack(images), label

def pad_collate_fn(batch):
    max_length = max([item[0].size(0) for item in batch])
    padded_batch = []
    labels = []
    for item, label in batch:
        padding = torch.zeros((max_length - item.size(0), *item.size()[1:]))
        padded_item = torch.cat((item, padding), dim=0)
        padded_batch.append(padded_item)
        labels.append(label)
    return torch.stack(padded_batch), torch.tensor(labels, dtype=torch.long)

# Parse arguments
parser = argparse.ArgumentParser(description='Sign Language Transformer Training')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use for computation')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# Dataset path and parameters
dataset_path = 'datasets'
batch_size = 8
num_epochs = 500
learning_rate = 0.001

# Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = SequenceDataset(root=dataset_path, transform=transform)
all_sequences = list(range(len(dataset)))
labels = dataset.labels
train_indices, test_indices = train_test_split(
    all_sequences, test_size=0.2, random_state=42, stratify=labels
)

# Subset datasets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn, num_workers=4, pin_memory=True)

# Model setup
num_classes = len(dataset.label_to_idx)
model = SignLanguageTransformer(input_dim=512, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
scaler = torch.cuda.amp.GradScaler()

# TensorBoard
writer = SummaryWriter('runs/sign_language')

# Training and evaluation
best_loss = float('inf')
early_stop_count = 0
early_stop_limit = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

    for sequences, labels in progress_bar:
        sequences, labels = sequences.to(device), labels.to(device)
        batch_size, sequence_length, channels, height, width = sequences.size()
        sequences = sequences.view(batch_size * sequence_length, channels, height, width)

        with torch.no_grad():
            features = feature_extractor(sequences).to(device)
        features = features.view(batch_size, sequence_length, -1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(features)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / len(progress_bar))

    avg_loss = running_loss / len(train_loader)
    writer.add_scalar('Loss/train', avg_loss, epoch)
    scheduler.step(avg_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            batch_size, sequence_length, channels, height, width = sequences.size()
            sequences = sequences.view(batch_size * sequence_length, channels, height, width)
            features = feature_extractor(sequences).to(device)
            features = features.view(batch_size, sequence_length, -1)

            outputs = model(features)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            all_preds.append(predictions.cpu())
            all_labels.append(labels.cpu())

    val_loss /= len(test_loader)
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    val_accuracy = (all_preds == all_labels).sum().item() / len(all_labels)

    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    if val_loss < best_loss:
        best_loss = val_loss
        early_stop_count = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        early_stop_count += 1
        if early_stop_count >= early_stop_limit:
            print("Early stopping triggered.")
            break

# Evaluation
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
all_preds, all_labels = [], []
test_loss = 0.0
correct_predictions = 0
with torch.no_grad():
    for sequences, labels in tqdm(test_loader, desc="Testing", unit="batch"):
        sequences, labels = sequences.to(device), labels.to(device)
        batch_size, sequence_length, channels, height, width = sequences.size()
        sequences = sequences.view(batch_size * sequence_length, channels, height, width)
        features = feature_extractor(sequences).to(device)
        features = features.view(batch_size, sequence_length, -1)

        outputs = model(features)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        predictions = outputs.argmax(dim=1)
        correct_predictions += (predictions == labels).sum().item()
        all_preds.append(predictions.cpu())
        all_labels.append(labels.cpu())

test_loss /= len(test_loader)
accuracy = correct_predictions / len(test_dataset)
all_preds = torch.cat(all_preds, dim=0)
all_labels = torch.cat(all_labels, dim=0)

print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}')
print(classification_report(all_labels, all_preds, target_names=list(dataset.label_to_idx.keys())))

# Save final model
model_path = "model/sign_language_transformer_final.pth"
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')
