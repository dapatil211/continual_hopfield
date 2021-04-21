def create_split_cifar100():
    pass


def create_split_mnist():
    pass


def create_permuted_mnist():
    pass


def get_dataloaders(name):
    if name == "split_cifar100":
        return create_split_cifar100()
    elif name == "split_mnist":
        return create_split_mnist()
    elif name == "permuted_mnist":
        return create_permuted_mnist()

    raise ValueError(f"Dataset {name} is not available")

