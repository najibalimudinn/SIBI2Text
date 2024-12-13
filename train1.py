import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from utils.signLanguageTransformer import SignLanguageTransformer
from utils.featureExtractor import feature_extractor

# Custom Dataset to handle sequences of images
class SequenceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.data = []

        # Group files by sequence
        for seq_name in sorted(os.listdir(root)):
            seq_path = os.path.join(root, seq_name)
            if os.path.isdir(seq_path):
                sequences = {}
                for file_name in sorted(os.listdir(seq_path)):
                    if file_name.endswith('.jpg'):
                        parts = file_name.split('__')
                        if len(parts) > 1:
                            seq_num = parts[1].split('.')[0]
                        else:
                            seq_num = 'default'
                        if seq_num not in sequences:
                            sequences[seq_num] = []
                        sequences[seq_num].append(os.path.join(seq_path, file_name))
                for seq_files in sequences.values():
                    self.data.append(seq_files)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        frames = self.data[idx]
        images = [Image.open(frame).convert('RGB') for frame in frames]
        if self.transform:
            images = [self.transform(img) for img in images]
        return torch.stack(images)  # Stack into (sequence_length, channels, height, width)

def pad_collate_fn(batch):
    # Find the maximum sequence length in the batch
    max_length = max([item.size(0) for item in batch])
    
    # Pad sequences to the maximum length
    padded_batch = []
    for item in batch:
        padding = torch.zeros((max_length - item.size(0), *item.size()[1:]))
        padded_item = torch.cat((item, padding), dim=0)
        padded_batch.append(padded_item)
    
    return torch.stack(padded_batch)

# Define dataset path and parameters
dataset_path = 'datasets'
batch_size = 4  # Adjust based on GPU/CPU memory
num_epochs = 20
learning_rate = 0.001

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load dataset
dataset = SequenceDataset(root=dataset_path, transform=transform)

# Split dataset into training and testing
all_sequences = list(range(len(dataset)))
train_indices, test_indices = train_test_split(all_sequences, test_size=0.2, random_state=42)

# Subset datasets
train_dataset = torch.utils.data.Subset(dataset, train_indices)
test_dataset = torch.utils.data.Subset(dataset, test_indices)

# Define DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)

# Get number of classes
num_classes = len(os.listdir(dataset_path))

# Initialize model, loss function, and optimizer
model = SignLanguageTransformer(input_dim=512, num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", unit="batch")

    for sequences in progress_bar:  # sequences.shape = (batch_size, sequence_length, channels, height, width)
        batch_size, sequence_length, channels, height, width = sequences.size()

        # Flatten sequences for feature extraction
        sequences = sequences.view(batch_size * sequence_length, channels, height, width)
        
        # Extract features
        with torch.no_grad():
            features = feature_extractor(sequences)  # features.shape = (batch_size * sequence_length, feature_dim)
        features = features.view(batch_size, sequence_length, -1)  # Reshape to (batch_size, sequence_length, feature_dim)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)  # outputs.shape = (batch_size, num_classes)
        labels = torch.arange(batch_size).long()  # Example labels (adjust based on your dataset)

        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / len(progress_bar))

    # Print epoch loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

# Evaluation loop
model.eval()
test_loss = 0.0
with torch.no_grad():
    for sequences in tqdm(test_loader, desc="Evaluating", unit="batch"):
        batch_size, sequence_length, channels, height, width = sequences.size()
        sequences = sequences.view(batch_size * sequence_length, channels, height, width)
        features = feature_extractor(sequences)
        features = features.view(batch_size, sequence_length, -1)

        outputs = model(features)
        labels = torch.arange(batch_size).long()  # Adjust based on your dataset
        loss = criterion(outputs, labels)
        test_loss += loss.item()

print(f"Test Loss: {test_loss/len(test_loader):.4f}")

# Save the trained model
model_path = "model/sign_language_transformer2.pth"
torch.save(model.state_dict(), model_path)
print(f'Model saved to {model_path}')