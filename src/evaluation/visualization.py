import matplotlib.pyplot as plt


def _show_tensor_image(image_tensor, cmap=None):
    plt.imshow(image_tensor.permute(1, 2, 0), cmap=cmap)
    plt.axis("off")


def show_dataset_samples(images, num_samples=6):
    plt.figure(figsize=(6, 6))
    for index in range(num_samples):
        plt.subplot(2, 3, index + 1)
        _show_tensor_image(images[index])
    plt.tight_layout()
    plt.show()


def show_original_and_edge(original_image, edge_image):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    _show_tensor_image(original_image)
    plt.title("Original")

    plt.subplot(1, 2, 2)
    _show_tensor_image(edge_image, cmap="gray")
    plt.title("Edge")
    plt.tight_layout()
    plt.show()


def show_edge_target_pair(edge_image, target_image, edge_title="Edge Input"):
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    _show_tensor_image(edge_image, cmap="gray")
    plt.title(edge_title)

    plt.subplot(1, 2, 2)
    _show_tensor_image(target_image)
    plt.title("Target Image")
    plt.tight_layout()
    plt.show()


def show_two_column_batch(edge_batch, image_batch, edge_title, image_title, num_samples=2):
    plt.figure(figsize=(6, 4))
    for index in range(num_samples):
        plt.subplot(2, 2, 2 * index + 1)
        _show_tensor_image(edge_batch[index], cmap="gray")
        plt.title(edge_title)

        plt.subplot(2, 2, 2 * index + 2)
        _show_tensor_image(image_batch[index])
        plt.title(image_title)
    plt.tight_layout()
    plt.show()


def plot_loss_curve(losses, title, ylabel, label=None):
    plt.figure(figsize=(5, 3))
    plt.plot(losses, marker="o", label=label)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    if label is not None:
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_gan_losses(generator_losses, discriminator_losses):
    plt.figure(figsize=(6, 3))
    plt.plot(generator_losses, marker="o", label="Generator Loss")
    plt.plot(discriminator_losses, marker="o", label="Discriminator Loss")
    plt.title("GAN Training Losses")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def show_prediction_triplets(edge_batch, output_batch, target_batch, output_title):
    plt.figure(figsize=(8, 6))
    for index in range(3):
        plt.subplot(3, 3, 3 * index + 1)
        _show_tensor_image(edge_batch[index], cmap="gray")
        plt.title("Edge")

        plt.subplot(3, 3, 3 * index + 2)
        _show_tensor_image(output_batch[index])
        plt.title(output_title)

        plt.subplot(3, 3, 3 * index + 3)
        _show_tensor_image(target_batch[index])
        plt.title("Target")
    plt.tight_layout()
    plt.show()


def show_model_comparison(edge_batch, cnn_output_batch, gan_output_batch, target_batch, edge_title):
    plt.figure(figsize=(10, 8))
    for index in range(3):
        plt.subplot(3, 4, 4 * index + 1)
        _show_tensor_image(edge_batch[index], cmap="gray")
        plt.title(edge_title)

        plt.subplot(3, 4, 4 * index + 2)
        _show_tensor_image(cnn_output_batch[index])
        plt.title("CNN")

        plt.subplot(3, 4, 4 * index + 3)
        _show_tensor_image(gan_output_batch[index])
        plt.title("GAN")

        plt.subplot(3, 4, 4 * index + 4)
        _show_tensor_image(target_batch[index])
        plt.title("Target")
    plt.tight_layout()
    plt.show()
