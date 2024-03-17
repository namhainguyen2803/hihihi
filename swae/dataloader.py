from pytorch_balanced_sampler import *
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader


class MNISTDataLoader:
    def __init__(self, data_dir="data/", train_batch_size=250, test_batch_size=250):
        self.test_loader = None
        self.train_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_classes = 10

        self.create_dataset()

    def create_dataset(self):
        train_set = datasets.MNIST(self.data_dir + "train/", train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       # transforms.Normalize((0.1307,), (0.3081,))
                                   ]))
        self.train_dataset = train_set

        test_set = datasets.MNIST(self.data_dir + "test/", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      # transforms.Normalize((0.1307,), (0.3081,))
                                  ]))
        self.test_dataset = test_set

    def create_dataloader(self):
        instances_indices = torch.arange(len(self.train_dataset.targets))
        all_classes_indices = list()
        for i in range(self.num_classes):
            class_index = instances_indices[torch.tensor(self.train_dataset.targets) == i].tolist()
            all_classes_indices.append(class_index)

        batch_sampler = SamplerFactory().get(
            class_idxs=all_classes_indices,
            batch_size=self.train_batch_size,
            n_batches=len(self.train_dataset.targets) // self.train_batch_size,
            alpha=1,
            kind='fixed'
        )

        self.train_loader = DataLoader(self.train_dataset, batch_sampler=batch_sampler)

        instances_indices = torch.arange(len(self.test_dataset.targets))
        all_classes_indices = list()
        for i in range(self.num_classes):
            class_index = instances_indices[torch.tensor(self.test_dataset.targets) == i].tolist()
            all_classes_indices.append(class_index)

        test_batch_sampler = SamplerFactory().get(
            class_idxs=all_classes_indices,
            batch_size=self.test_batch_size,
            n_batches=len(self.test_dataset.targets) // self.test_batch_size,
            alpha=1,
            kind='fixed'
        )

        self.test_loader = DataLoader(self.test_dataset, batch_sampler=test_batch_sampler)

        # self.test_loader = DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False)

        return self.train_loader, self.test_loader


class CIFAR10DataLoader:
    def __init__(self, data_dir="data/", train_batch_size=80, test_batch_size=80):
        self.test_loader = None
        self.train_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.data_dir = data_dir
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_classes = 10

        self.create_dataset()

    def create_dataset(self):

        train_set = datasets.CIFAR10(self.data_dir + "train/",
                                     train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomCrop(32, padding=4),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                     ]))

        test_set = datasets.CIFAR10(self.data_dir + "test/",
                                    train=False, download=True,
                                    transform=transforms.Compose([
                                        transforms.ToTensor()
                                    ]))

        self.train_dataset = train_set
        self.test_dataset = test_set

    def create_dataloader(self):

        instances_indices = torch.arange(len(self.train_dataset.targets))
        all_classes_indices = list()
        for i in range(self.num_classes):
            class_index = instances_indices[torch.tensor(self.train_dataset.targets) == i].tolist()
            all_classes_indices.append(class_index)

        batch_sampler = SamplerFactory().get(
            class_idxs=all_classes_indices,
            batch_size=self.train_batch_size,
            n_batches=len(self.train_dataset.targets) // self.train_batch_size,
            alpha=1,
            kind='fixed'
        )
        self.train_loader = DataLoader(self.train_dataset, batch_sampler=batch_sampler)

        instances_indices = torch.arange(len(self.test_dataset.targets))
        all_classes_indices = list()
        for i in range(self.num_classes):
            class_index = instances_indices[torch.tensor(self.test_dataset.targets) == i].tolist()
            all_classes_indices.append(class_index)

        test_batch_sampler = SamplerFactory().get(
            class_idxs=all_classes_indices,
            batch_size=self.test_batch_size,
            n_batches=len(self.test_dataset.targets) // self.test_batch_size,
            alpha=1,
            kind='fixed'
        )
        self.test_loader = DataLoader(self.test_dataset, batch_sampler=test_batch_sampler)

        return self.train_loader, self.test_loader
