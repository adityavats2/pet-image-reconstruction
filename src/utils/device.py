import torch


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_summary() -> torch.device:
    has_cuda = torch.cuda.is_available()
    print(has_cuda)
    print(torch.cuda.get_device_name(0) if has_cuda else "No GPU")
    device = get_device()
    print("Device:", device)
    return device
