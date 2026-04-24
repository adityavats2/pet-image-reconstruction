import torch


def train_gan(
    generator,
    discriminator,
    train_loader,
    adversarial_loss,
    reconstruction_loss,
    g_optimizer,
    d_optimizer,
    device,
    epochs=3,
    lambda_l1=100,
):
    g_losses = []
    d_losses = []

    for epoch in range(epochs):
        generator.train()
        discriminator.train()

        running_g_loss = 0.0
        running_d_loss = 0.0

        for edges, real_images in train_loader:
            edges = edges.to(device)
            real_images = real_images.to(device)

            batch_size = edges.size(0)
            real_labels = torch.ones((batch_size, 1, 1, 1), device=device)
            fake_labels = torch.zeros((batch_size, 1, 1, 1), device=device)

            d_optimizer.zero_grad()
            fake_images = generator(edges).detach()

            real_preds = discriminator(edges, real_images)
            fake_preds = discriminator(edges, fake_images)

            d_real_loss = adversarial_loss(real_preds, real_labels)
            d_fake_loss = adversarial_loss(fake_preds, fake_labels)
            d_loss = (d_real_loss + d_fake_loss) / 2

            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            fake_images = generator(edges)
            fake_preds = discriminator(edges, fake_images)

            g_adv_loss = adversarial_loss(fake_preds, real_labels)
            g_l1_loss = reconstruction_loss(fake_images, real_images)
            g_loss = g_adv_loss + lambda_l1 * g_l1_loss

            g_loss.backward()
            g_optimizer.step()

            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()

        avg_d_loss = running_d_loss / len(train_loader)
        avg_g_loss = running_g_loss / len(train_loader)

        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

        print(
            f"Epoch [{epoch+1}/{epochs}] - "
            f"D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}"
        )

    return g_losses, d_losses
