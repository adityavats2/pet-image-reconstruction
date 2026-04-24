def train_cnn(model, train_loader, criterion, optimizer, device, epochs=3):
    model.train()
    train_losses = []

    for epoch in range(epochs):
        running_loss = 0.0

        for edges, targets in train_loader:
            edges = edges.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(edges)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

    return train_losses
