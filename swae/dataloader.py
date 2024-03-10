from pytorch_balanced_sampler import *
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader


class MNISTDataLoader():
    def __init__(self, data_dir="data/", train_batch_size=64, test_batch_size=64):
        self.test_loader = None
        self.train_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.create_dataset()
        self.num_classes = 10

    def create_dataset(self):
        train_set = datasets.MNIST(self.data_dir + "train/", train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
        self.train_dataset = train_set

        test_set = datasets.MNIST(self.data_dir + "test/", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
        self.test_dataset = test_set

    def create_train_dataloader(self):
        instances_indices = torch.arange(len(self.train_dataset.targets))
        all_classes_indices = list()
        for i in range(self.num_classes):
            class_index = instances_indices[self.train_dataset.targets == i]
            all_classes_indices.append(class_index)

        batch_sampler = SamplerFactory().get(
            class_idxs=all_classes_indices,
            batch_size=self.train_batch_size,
            n_batches=250,
            alpha=1,
            kind='fixed'
        )

        self.train_loader = DataLoader(self.train_dataset, batch_sampler=batch_sampler)

        self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)

        return self.train_loader, self.test_loader
