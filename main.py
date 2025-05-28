import argparse
import yaml
import json
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models.HoViT import HoViT

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def load_index_dict(path):
    with open(path, "r") as f:
        return json.load(f)

def get_dataloaders(config, index_dict, batch_size):
    # transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(root=config['image_root'], transform=transform)

    train_data = Subset(dataset, index_dict['train_idx'])
    val_data = Subset(dataset, index_dict['val_idx'])
    test_data = Subset(dataset, index_dict['test_idx'])

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader

def train(model, train_loader, val_loader, device, config):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 1e-4))
    epochs = config.get('epochs', 10)

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {total_loss:.4f}")

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"[Evaluation] Accuracy: {100 * correct / total:.2f}%")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/HoViT.yaml')
    parser.add_argument('--index_dict', type=str, default='configs/index_dict.json')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train')
    parser.add_argument('--checkpoint', type=str, default='pretrained/BaseLine_HoViT_44.pth')
    args = parser.parse_args()

    config = load_config(args.config)
    index_dict = load_index_dict(args.index_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = HoViT(config).to(device)
    train_loader, val_loader, test_loader = get_dataloaders(config, index_dict, batch_size=config.get('batch_size', 32))

    if args.mode == 'eval':
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        evaluate(model, test_loader, device)
    else:
        model.train()
        train(model, train_loader, val_loader, device, config)
        os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint)
        print(f"Model saved to {args.checkpoint}")

if __name__ == "__main__":
    main()