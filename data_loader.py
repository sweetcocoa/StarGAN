from torch.utils import data
from torchvision import transforms
from torchvision import datasets
import torch

def get_loader(config):
    """Builds and returns Dataloader for MNIST and SVHN dataset."""

    transform = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    mnist = datasets.MNIST(root=config.mnist_path, download=True, transform=transform)
    svhn = datasets.SVHN(root=config.svhn_path, download=True, transform=transform)



    mnist_loader = torch.utils.data.DataLoader(dataset=mnist,
                                               batch_size=config.batch_size,
                                               shuffle=True,
                                               num_workers=config.num_workers)

    svhn_loader = torch.utils.data.DataLoader(dataset=svhn,
                                              batch_size=config.batch_size,
                                              shuffle=True,
                                              num_workers=config.num_workers)

    return svhn_loader, mnist_loader


if __name__ == "__main__":
    class Config:
        batch_size = 4
        num_workers = 4
        svhn_path = "svhn"
        mnist_path = "mnist"
        image_size = 28
    print("config", Config().num_workers)
    get_loader(Config())