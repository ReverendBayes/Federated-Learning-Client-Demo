#!/usr/bin/env python3
"""
flclient.py: Federated Learning Client Demo (FedAvg)

Usage:
    python flclient.py \
        --data mnist \
        --model cnn \
        --server http://localhost:8080 \
        --rounds 1 \
        --epochs 1 \
        --batch-size 32 \
        --lr 0.01 \
        --metrics-out metrics.json

Features:
    - Built-in MNIST dataset loader
    - Simple CNN model
    - Configurable rounds, epochs, batch size, learning rate
    - Fetches initial global parameters (if server supports) via GET /get_parameters
    - Posts local metrics via POST /submit_metrics
    - Saves local metrics JSON to disk
Dependencies:
    torch, torchvision, requests
"""
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import requests

# --- Model Definition ---
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.dropout2 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# --- Training & Evaluation ---
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return correct / len(test_loader)

# --- Federated Client Logic ---
def fetch_global_parameters(server_url):
    try:
        r = requests.get(f"{server_url.rstrip('/')}/get_parameters")
        r.raise_for_status()
        params = r.json().get('parameters', {})
        return {k: torch.tensor(v) for k, v in params.items()}
    except Exception as e:
        print(f"Could not fetch global parameters: {e}")
        return {}

def post_metrics(server_url, metrics):
    try:
        r = requests.post(f"{server_url.rstrip('/')}/submit_metrics", json={"metrics": metrics})
        r.raise_for_status()
        print(f"Server response: {r.text}")
    except Exception as e:
        print(f"Failed to post metrics: {e}")

def apply_parameters(model, param_dict):
    sd = model.state_dict()
    for k, v in param_dict.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v
    model.load_state_dict(sd)

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client Demo (FedAvg)")
    parser.add_argument("--data", choices=["mnist"], default="mnist", help="Dataset to use")
    parser.add_argument("--model", choices=["cnn"], default="cnn", help="Model architecture")
    parser.add_argument("--server", required=True, help="Federation server URL")
    parser.add_argument("--rounds", type=int, default=1, help="Federation rounds")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per round")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--metrics-out", default="metrics.json", help="Path to save local metrics")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA if available")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = CNN().to(device)

    # Fetch and apply global params
    global_params = fetch_global_parameters(args.server)
    if global_params:
        apply_parameters(model, global_params)

    # Data Loaders
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    # Federated rounds
    metrics = {"rounds": []}
    for r in range(1, args.rounds + 1):
        print(f"Round {r}/{args.rounds}")
        avg_loss = train(model, device, train_loader, optimizer, r)
        acc = test(model, device, test_loader)
        print(f"  Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        metrics["rounds"].append({"round": r, "loss": avg_loss, "accuracy": acc})

    # Save metrics locally and post to server
    with open(args.metrics_out, "w") as mf:
        json.dump(metrics, mf, indent=2)
    print(f"Local metrics saved to {args.metrics_out}")

    post_metrics(args.server, metrics)

if __name__ == "__main__":
    main()
