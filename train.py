import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.io import read_image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.featureExtractor import FeatureExtractor
from tqdm import tqdm

# Dataset Class
class MotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = []  # Menyimpan path file dan label

        # Load dataset structure
        for label, class_dir in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.endswith(('.jpg', '.png', '.jpeg')):
                        img_path = os.path.join(class_path, img_file)
                        self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = read_image(img_path).float() / 255.0  # Normalize to [0, 1]
        if self.transform:
            image = self.transform(image)
        return image, label

# Training Function
def train_model(data_dir, output_dir, num_classes, batch_size=16, num_epochs=20, learning_rate=1e-3):
    # Load Feature Extractor
    feature_extractor = FeatureExtractor()

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize image to match ResNet input
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ])
    dataset = MotionDataset(data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, Optimizer, and Loss Function
    model = SignLanguageTransformer(input_dim=512, num_classes=num_classes).to('cuda')
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            # Extract Features
            images = images.to('cuda')
            features = feature_extractor(images)

            # Forward Pass
            labels = labels.to('cuda')
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
    # Configuration
    data_directory = "datasets"  # Path to the dataset
    output_directory = "output_models"  # Path to save model checkpoints
    num_classes = len(os.listdir(data_directory))  # Number of classes (folders)
    os.makedirs(output_directory, exist_ok=True)

    # Train the Model
    train_model(data_directory, output_directory, num_classes)
