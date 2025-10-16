import copy

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Dataset
from torchvision import datasets
from torchvision.models import VisionTransformer, ViT_B_32_Weights, ViT_B_16_Weights
from torchvision import transforms

from bottleneck import Bottleneck
from bottleneck_vision_transformer import BottleneckVisionTransformer

from torch.utils.data import Dataset


class EmptyDataset(Dataset):
    def __init__(self):
        self.data = []

    def __len__(self):
        return 0

    def __getitem__(self, idx):
        raise IndexError("This dataset is empty")

def split_dataset(dataset, val_fraction=0.2):

    if val_fraction <= 0.0: return dataset, EmptyDataset()

    train_size = int((1 - val_fraction) * len(dataset))  # 90% train
    val_size = len(dataset) - train_size   # 10% val

    from main import seed
    # Deterministic split
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return train_dataset, val_dataset

def prepare_dataset(params):


    if params.dataset == 'CIFAR100':
        # Load CIFAR-100 dataset
        # Data augmentation and normalization
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408],
                                 std=[0.2675, 0.2565, 0.2761])
        ])

        print("Loading CIFAR-100 dataset...")
        train_dataset_full = datasets.CIFAR100(root='./data', train=True,
                                               download=True)

        test_dataset = datasets.CIFAR100(root='./data', train=False,
                                         download=True, transform=test_transform)

        train_dataset2, val_dataset = split_dataset(train_dataset_full, params.val_fraction)
        val_dataset.dataset.transform = test_transform


    elif params.dataset == 'CIFAR10':
        print("Loading CIFAR-10 dataset...")
        # Data augmentation and normalization
        train_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
        ])

        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2470, 0.2435, 0.2616])
        ])

        train_dataset_full = datasets.CIFAR10(root='./data', train=True, download=True)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

        train_dataset2, val_dataset = split_dataset(train_dataset_full, params.val_fraction)
        val_dataset.dataset.transform = test_transform

    elif params.dataset == "TinyImageNet":
        print("Loading TinyImageNet dataset...")
        # ImageNet mean and std (for pretrained ViT normalization)
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        # --- Data augmentation and normalization ---
        train_transform = transforms.Compose([
            transforms.Resize(256),  # Resize shorter side to 256
            transforms.RandomCrop(224),  # Random crop to 224×224
            transforms.RandomHorizontalFlip(),  # Data augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        # --- Datasets ---
        data_dir = './data/tiny-imagenet-200'

        train_dataset_full = datasets.ImageFolder(root=f'{data_dir}/train', transform=train_transform)
        test_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=test_transform)
        # test_dataset = datasets.ImageFolder(root=f'{data_dir}/test', transform=test_transform)

        train_dataset2, val_dataset = split_dataset(train_dataset_full, params.val_fraction)
        val_dataset.dataset.transform = test_transform

        print(f"Tiny ImageNet loaded: {len(train_dataset2)} train, {len(val_dataset)} val samples.")

    elif params.dataset == "CalTech256":
        print("Loading CalTech256 dataset...")
        # ImageNet mean and std (re-used for consistency with TinyImageNet)
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        # --- Data augmentation and normalization (re-using TinyImageNet transforms) ---
        train_transform = transforms.Compose([
            transforms.Resize(256),  # Resize shorter side to 256
            transforms.RandomCrop(224),  # Random crop to 224×224
            transforms.RandomHorizontalFlip(),  # Data augmentation
            # CalTech256 contains some grayscale images, so we convert to RGB before ToTensor
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            # CalTech256 contains some grayscale images, so we convert to RGB before ToTensor
            transforms.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])

        # --- Datasets ---
        data_dir = '/data/users/vtsouval/torch_datasets/caltech256'

        # Caltech256 loads the *entire* dataset. Download=True ensures it's available.
        full_dataset = datasets.Caltech256(root=f'{data_dir}', download=False, transform=train_transform)

        # The dataset doesn't have official splits, so we'll create train, val, and test splits
        # Split into a 'training' portion and a 'test' portion (e.g., 80% train / 20% test)
        train_val_dataset, test_dataset_temp = split_dataset(full_dataset, 0.2)

        # NOTE: CalTech256 is not intrinsically split, so we must manually assign the test_transform to the test split's underlying dataset
        test_dataset_temp.dataset.transform = test_transform
        test_dataset = test_dataset_temp  # Rename for consistency

        # Split the remaining 'training' portion into train and validation (e.g., 90% train / 10% val of the 80%)
        train_dataset2, val_dataset_temp = split_dataset(train_val_dataset, params.val_fraction)

        # NOTE: Must reassign transform on the validation split's underlying dataset
        val_dataset_temp.dataset.transform = test_transform
        val_dataset = val_dataset_temp  # Rename for consistency

        print(
            f"CalTech256 loaded and split: {len(train_dataset2)} train, {len(val_dataset)} val, {len(test_dataset)} test samples.")
    elif params.dataset == "Food101":
        print("Loading Food-101 dataset...")

        from torchvision.datasets import Food101

        # Mean and std for Food-101 (computed from dataset)
        food101_mean = [0.5450, 0.4436, 0.3435]
        food101_std = [0.2520, 0.2605, 0.2728]

        # --- Data augmentation and normalization ---
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=food101_mean, std=food101_std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=food101_mean, std=food101_std)
        ])

        # --- Datasets ---
        data_dir = './data'

        train_dataset_full = Food101(root=data_dir, split="train", transform=train_transform, download=True)
        test_dataset = Food101(root=data_dir, split="test", transform=test_transform, download=True)

        # --- Split train into train/val ---
        train_dataset2, val_dataset = split_dataset(train_dataset_full, params.val_fraction)
        val_dataset.dataset.transform = test_transform

        print(f"Food-101 loaded: {len(train_dataset2)} train, {len(val_dataset)} val samples.")


    else:
        print("unknown dataset!!")

    if params.pre_train_fraction > 0.0:
        train_dataset, pretrain_dataset = split_dataset(train_dataset2, params.pre_train_fraction)
        pretrain_dataset, pretrain_val_dataset = split_dataset(pretrain_dataset, val_fraction=0.1)

        pretrain_dataset.dataset.transform = train_transform
        pretrain_val_dataset.dataset.transform = test_transform

        pretrain_loader = DataLoader(pretrain_dataset, batch_size=params.batch_size, shuffle=True)
        pretrain_val_loader = DataLoader(pretrain_val_dataset, batch_size=params.batch_size, shuffle=False)
    else:
        train_dataset = train_dataset2
        pretrain_loader = None
        pretrain_val_loader = None

    train_dataset.dataset.transform = train_transform

    # test_dataset.dataset.transform = test_transform


    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=16, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False, num_workers=16, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False , num_workers=8, pin_memory=True, persistent_workers=True)


    return train_loader, val_loader, test_loader, pretrain_loader, pretrain_val_loader

def prepare_original_model(num_classes, device, parallel=False, patch_size=16):
    # Load pre-trained Vision Transformer B/32 (4x faster than B/16)
    print(f"\nLoading pre-trained Vision Transformer (ViT-B/{16})...")
    # original_model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    original_model = VisionTransformer(image_size=224,
                                       patch_size=patch_size,
                                       num_layers=12,
                                       num_heads=12,
                                       hidden_dim=768,
                                       mlp_dim=3072,
                                       num_classes=1000, )

    if patch_size == 32:
        original_model.load_state_dict(ViT_B_32_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True),
                                       strict=True)
    elif patch_size == 16:
        original_model.load_state_dict(ViT_B_16_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True),
                                       strict=True)


    # Modify the classification head for CIFAR-100 (100 classes)
    # original_model.heads = nn.Linear(original_model.heads[0].in_features, NUM_CLASSES)

    original_model.heads = nn.Linear(original_model.heads[0].in_features, num_classes)

    if parallel:
        original_model = nn.DataParallel(original_model)

    original_model.to(device)
    m = original_model.module if hasattr(original_model, "module") else original_model
    # svhn, food101, cars 196, fashion mnist, euro sat
    return m

def prepare_bottleneck_model(num_classes, bottleneck_dim, path, device, parallel=False, patch_size=16):
    # 1. Recreate architecture
    bottleneck = Bottleneck(embedding_dim=768, bottleneck_dim=bottleneck_dim)

    # 2. Load pretrained weights
    saved_model = torch.load(path, map_location=device)

    bottleneck.load_state_dict(saved_model['model_state_dict'])

    # 3. Create a deep copy (so each ViT has its own independent instance)
    bottleneck_copy = copy.deepcopy(bottleneck)

    bottleneck_model = BottleneckVisionTransformer(bottleneck_copy,
                                                   image_size=224,
                                                   patch_size=patch_size,
                                                   num_layers=12,
                                                   num_heads=12,
                                                   hidden_dim=768,
                                                   mlp_dim=3072,
                                                   num_classes=1000,
                                                   )

    if patch_size == 32:
        bottleneck_model.load_pretrained_weights(ViT_B_32_Weights.IMAGENET1K_V1)
    elif patch_size == 16:
        bottleneck_model.load_pretrained_weights(ViT_B_16_Weights.IMAGENET1K_V1)
    # bottleneck_model.load_state_dict(ViT_B_32_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True))
    # bottleneck_model.heads = nn.Linear(bottleneck_model.heads[0].in_features, NUM_CLASSES)
    bottleneck_model.heads = nn.Linear(bottleneck_model.heads[0].in_features, num_classes)

    if parallel:
        bottleneck_model = nn.DataParallel(bottleneck_model)

    bottleneck_model = bottleneck_model.to(device)

    m = bottleneck_model.module if hasattr(bottleneck_model, "module") else bottleneck_model

    return m