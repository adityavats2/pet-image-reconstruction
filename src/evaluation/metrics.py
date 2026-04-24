import torch
import torch.nn as nn


def evaluate_l1(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    count = 0
    criterion = nn.L1Loss()

    with torch.no_grad():
        for edges, targets in dataloader:
            edges = edges.to(device)
            targets = targets.to(device)

            outputs = model(edges)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            count += 1

    return total_loss / count
