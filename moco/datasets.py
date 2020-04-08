import os
import itertools

import torch.utils.data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

import moco.loader


class InfiniteSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source):
        super(InfiniteSampler, self).__init__(data_source)
        self.data_source = data_source

    @property
    def num_samples(self):
        return 2 ** 48

    def __iter__(self):
        n = len(self.data_source)

        def inf():
            while True:
                yield iter(torch.randperm(n).tolist())

        return itertools.chain.from_iterable(inf())

    def __len__(self):
        return self.num_samples



def load_moco_train_set(dataset_name, data_dir, aug_plus):
    # Data loading code
    if dataset_name == 'imagenet':
        train_dir = os.path.join(data_dir, 'train')
        if not os.path.exists(train_dir):
            train_dir = data_dir
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if aug_plus:
            # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
            augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]
        else:
            # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
            augmentation = [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ]

        return datasets.ImageFolder(
            train_dir,
            moco.loader.TwoCropsTransform(transforms.Compose(augmentation)))

    elif dataset_name == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])
        augmentation = [
            transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            normalize
        ]

        return datasets.CIFAR10(
            data_dir, train=True,
            transform=moco.loader.TwoCropsTransform(transforms.Compose(augmentation)),
            download=True)

    else:
        raise ValueError('Unknown dataset {}'.format(dataset_name))


def load_clf_train_val_set(dataset_name, data_dir):
    if dataset_name == 'imagenet':
        # Data loading code
        traindir = os.path.join(data_dir, 'train')
        valdir = os.path.join(data_dir, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

        return train_dataset, val_dataset

    elif dataset_name == 'cifar10':
        normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                         std=[0.2470, 0.2435, 0.2616])

        train_dataset = datasets.CIFAR10(
            data_dir, train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(size=32,
                                      padding=4,
                                      padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
            download=True)

        val_dataset = datasets.CIFAR10(
            data_dir, train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            download=True)

        return train_dataset, val_dataset
    else:
        raise ValueError('Unknown dataset {}'.format(dataset_name))


