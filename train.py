import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.featureExtractor import FeatureExtractor
from utils.signLanguageTransformer import SignLanguageTransformer
from tqdm import tqdm
import argparse
import json

# Dataset Class
class MotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []  # Store sequences of image paths and labels

        # Load dataset structure
        for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                sequences = {}
                for img_file in sorted(os.listdir(class_path)):
                    if img_file.endswith(('.jpg')):
                        if '__' in img_file:
                            base_name = img_file.split('__')[1].split('.jpg')[0]
                        else:
                            base_name = "normal"
                        if base_name not in sequences:
                            sequences[base_name] = []
                        sequences[base_name].append(os.path.join(class_path, img_file))
                for seq_files in sequences.values():
                    # Ensure each sequence has 10 images (0 to 9) and their augmentations
                    if len(seq_files) >= 10:
                        self.data.append((seq_files, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_paths, label = self.data[idx]
        images = [read_image(img_path).float() / 255.0 for img_path in img_paths]  # Normalize to [0, 1]
        if self.transform:
            images = [self.transform(image) for image in images]
        return torch.stack(images), label  # Return sequence of images and label

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

# Training Function
def train_model(data_dir, output_dir, num_classes, device, batch_size=16, num_epochs=20, learning_rate=1e-3):
    # Load Feature Extractor
    feature_extractor = FeatureExtractor()
    feature_extractor.to(device)

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match ResNet input
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    dataset = MotionDataset(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

    # Model, Optimizer, and Loss Function
    model = SignLanguageTransformer(input_dim=512, num_classes=num_classes).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for sequences, labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            sequences, labels = sequences.to(device), labels.to(device)
            batch_size, sequence_length, channels, height, width = sequences.size()
            sequences = sequences.view(batch_size * sequence_length, channels, height, width)

            # Extract Features
            with torch.no_grad():
                features = feature_extractor(sequences)
            features = features.view(batch_size, sequence_length, -1)

            # Forward Pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        # Scheduler Step
        scheduler.step(epoch_loss)

        # Logging
        print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {100 * correct / total:.2f}%")

        # Save Model Checkpoint
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch+1}.pth"))

if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description="Train a sign language recognition model.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for training (cpu or cuda)")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of epochs to train the model")
    args = parser.parse_args()

    # Configuration
    data_directory = "datasets"  # Path to the dataset
    output_directory = "model"  # Path to save model checkpoints
    num_classes = len([name for name in os.listdir(data_directory) if os.path.isdir(os.path.join(data_directory, name))])  # Number of classes (folders)
    os.makedirs(output_directory, exist_ok=True)

    # Train the Model
    train_model(data_directory, output_directory, num_classes, device=args.device, num_epochs=args.num_epochs)