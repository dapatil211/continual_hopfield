import os
import json
import random
import torch
import torchvision
from torchvision import transforms
import numpy as np


def create_logit_masks(task_split, num_classes):
    task_split = np.array(task_split)
    logit_mask = np.zeros((len(task_split), num_classes))
    for i in range(len(task_split)):
        logit_mask[i, task_split[i]] = 1.0
    return logit_mask


class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, filter_func, transform=None, metainfo_func=None):
        super().__init__()
        self.indices = []
        self.dataset = dataset
        self.filter_func = filter_func
        self.transform = transform
        if metainfo_func is None:
            metainfo_func = lambda dataset, i: dataset[i]
        self.metainfo_func = metainfo_func
        self.filter()

    def filter(self):
        for i in range(len(self.dataset)):
            if self.filter_func(self.metainfo_func(self.dataset, i)):
                self.indices.append(i)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        x = self.dataset[self.indices[index]]
        if self.transform is not None:
            x = self.transform(x)
        return x

def create_task_transform(task_id):
    return transforms.Lambda(lambda x: (x[0], x[1], task_id))

def create_split_cifar100(args):
    if args.cifar_split is not None and os.path.exists(args.cifar_split):
        with open(args.cifar_split) as f:
            task_split = json.load(f)
    else:
        classes = list(range(100))
        random.shuffle(classes)
        task_split = [sorted(classes[i * 5 : (i + 1) * 5]) for i in range(20)]
        with open(args.cifar_split, "w") as f:
            json.dump(task_split, f)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5071, 0.4865, 0.4409), std=(0.2673, 0.2564, 0.2762)
            ),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR100(
        "data/cifar100", download=True, transform=transform, train=True
    )
    test_dataset = torchvision.datasets.CIFAR100(
        "data/cifar100", download=True, transform=transform, train=False
    )
    split_train_dataloaders = []
    split_test_dataloaders = []
    if args.cross_validation:
        task_split = task_split[:3]
    else:
        task_split = task_split[3:]
    
    for task_id, task in enumerate(task_split):
        task_train_dataset = FilteredDataset(
            train_dataset,
            lambda x: x in task,
            transform=create_task_transform(task_id),
            metainfo_func=lambda dataset, i: dataset.targets[i],
        )
        split_train_dataloaders.append(
            torch.utils.data.DataLoader(
                task_train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
        )
        task_test_dataset = FilteredDataset(
            test_dataset,
            lambda x: x in task,
            transform=create_task_transform(task_id),
            metainfo_func=lambda dataset, i: dataset.targets[i],
        )
        split_test_dataloaders.append(
            torch.utils.data.DataLoader(
                task_test_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True,
            )
        )

    if args.same_head:
        num_classes = 5
        logit_mask = np.ones((len(task_split), num_classes))
    else:
        num_classes = 100
        logit_mask = create_logit_masks(task_split, num_classes)
    return (
        split_train_dataloaders,
        split_test_dataloaders,
        num_classes,
        logit_mask,
    )


def create_split_mnist():
    pass


def create_permuted_mnist():
    pass


def get_dataloaders(args):
    if args.dataset_name == "split_cifar100":
        return create_split_cifar100(args)
    elif args.dataset_name == "split_mnist":
        return create_split_mnist(args)
    elif args.dataset_name == "permuted_mnist":
        return create_permuted_mnist(args)

    raise ValueError(f"Dataset {name} is not available")

