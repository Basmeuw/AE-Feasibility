import copy
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from numpy.f2py.auxfuncs import throw_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from torchvision.models import vit_b_32, ViT_B_32_Weights, VisionTransformer, ViT_B_16_Weights

from tqdm import tqdm
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np
from params import Experiment

from bottleneck import Bottleneck
from bottleneck_vision_transformer import BottleneckVisionTransformer
import matplotlib.pyplot as plt


# Training function
def  train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': running_loss / len(pbar),
                          'acc': 100. * correct / total})

    return running_loss / len(loader), 100. * correct / total


# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100. * correct / total


def train(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, params, device, NUM_EPOCHS):
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # Training loop
    print("\nStarting training...\n")
    best_loss = float("inf")
    best_acc = 0.0

    pre_train_epochs = 0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

        # If it is the original model, skip
        if params.bottleneck_path is not None:
            # Pre-train epoch logic
            if epoch < pre_train_epochs:
                model.freeze_except_bottleneck()
            else:
                model.unfreeze()

        # Train and Evaluate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)



        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Step the scheduler
        scheduler.step()

        # Print info
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")

        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, f'models/temp/{params.title}.pth')
            print(f"✓ Saved best model with loss: {val_loss:.4f}, acc: {val_acc:.2f}%\n")

            # plot_metrics(train_losses, val_losses, , val_accuracies)

    print(f"\nTraining completed! Best validation loss: {best_loss:.4f} , acc: {best_acc:.2f}%")

    # Load and evaluate best model
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(f'models/temp/{params.title}.pth')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {final_acc:.2f}%")

    # Return collected data
    return {
        'final_test_accuracy': final_acc,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
    }, checkpoint



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

        train_dataset2, val_dataset = split_dataset(train_dataset_full, 0.1)
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

        train_dataset2, val_dataset = split_dataset(train_dataset_full, 0.1)
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

        train_dataset2, val_dataset = split_dataset(train_dataset_full, 0.1)
        val_dataset.dataset.transform = test_transform

        print(f"Tiny ImageNet loaded: {len(train_dataset2)} train, {len(val_dataset)} val samples.")

    else:
        throw_error("unknown dataset!!")


    train_dataset, pretrain_dataset = split_dataset(train_dataset2, 0.2)

    pretrain_dataset, pretrain_val_dataset = split_dataset(pretrain_dataset, val_fraction=0.1)

    train_dataset.dataset.transform = train_transform

    # test_dataset.dataset.transform = test_transform
    pretrain_dataset.dataset.transform = train_transform
    pretrain_val_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False )
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=params.batch_size, shuffle=True)
    pretrain_val_loader = DataLoader(pretrain_val_dataset, batch_size=params.batch_size, shuffle=False)


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

def prepare_models(num_classes, bottleneck_dim, bottleneck_path, device):
    print("\nLoading pre-trained Vision Transformer (ViT-B/32)...")
    # original_model = vit_b_32(weights=ViT_B_32_Weights.IMAGENET1K_V1)
    original_model = VisionTransformer(image_size=224,
                                       patch_size=32,
                                       num_layers=12,
                                       num_heads=12,
                                       hidden_dim=768,
                                       mlp_dim=3072,
                                       num_classes=1000, )

    original_model.load_state_dict(ViT_B_32_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True),
                                   strict=True)

    # Modify the classification head for CIFAR-100 (100 classes)
    # original_model.heads = nn.Linear(original_model.heads[0].in_features, NUM_CLASSES)
    original_model.to(device)

    # 1. Recreate architecture
    bottleneck = Bottleneck(embedding_dim=768, bottleneck_dim=bottleneck_dim)

    # 2. Load pretrained weights
    saved_model = torch.load(bottleneck_path, map_location=device)

    bottleneck.load_state_dict(saved_model['model_state_dict'])

    # 3. Create a deep copy (so each ViT has its own independent instance)
    bottleneck_copy = copy.deepcopy(bottleneck)

    bottleneck_model = BottleneckVisionTransformer(bottleneck_copy,
                                                   image_size=224,
                                                   patch_size=32,
                                                   num_layers=12,
                                                   num_heads=12,
                                                   hidden_dim=768,
                                                   mlp_dim=3072,
                                                   num_classes=1000,
                                                   )

    bottleneck_model.load_pretrained_weights(ViT_B_32_Weights.IMAGENET1K_V1)
    # bottleneck_model.load_state_dict(ViT_B_32_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True))
    # bottleneck_model.heads = nn.Linear(bottleneck_model.heads[0].in_features, NUM_CLASSES)
    bottleneck_model = bottleneck_model.to(device)
    # 1. Create ONE new classification head
    new_head = nn.Linear(original_model.heads[0].in_features, num_classes)

    # 2. Assign the EXACT SAME head to both models
    original_model.heads = new_head
    bottleneck_model.heads = new_head  # Now they are identical

    return original_model, bottleneck_model



# wrapper function that takes care of storing the data
def train_and_plot(model, criterion, optimizer, scheduler, params, device):

    # Convert Params instance to dict
    params_dict = params.__dict__

    is_bottleneck_model = params.bottleneck_path is not None

    train_loader, val_loader, test_loader, pretrain_loader, pretrain_val_loader = prepare_dataset(params)

    if params.pre_train:
        if is_bottleneck_model:
            print("Freezing BN for pretraining")
            model.freeze_except_bottleneck() # also unfreezes head
            pre_train_optimizer = optim.AdamW(model.parameters(), lr=params.pre_train_lr, weight_decay=params.weight_decay)
            pre_train_scheduler = optim.lr_scheduler.ConstantLR(pre_train_optimizer, factor=1.)

            pre_train_data, _ = train(model, pretrain_loader, pretrain_val_loader, pretrain_val_loader, criterion,
                                   pre_train_optimizer, pre_train_scheduler, params, device, params.pre_train_epochs)
            print("Pretraining finished, unfreezing model body now. Might need to switch back LR of BN model here")
            model.unfreeze()
        else:
            pass # TODO

    # Training
    training_data, final_model_checkpoint = train(
        model,
        train_loader,
        val_loader,
        test_loader,
        criterion,
        optimizer,
        scheduler,
        params,
        device,
        params.epochs
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    folder_path = 'runs/run_{}_{}'.format(timestamp, params.title)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'figures'), exist_ok=True)
    figure_path = os.path.join(folder_path, 'figures')
    # Dump to JSON file
    with open(folder_path + '/params.json', 'w') as f:
        json.dump(params_dict, f, indent=4)

    # Plot Training Loss vs Validation Loss
    plot_metrics(train_data=training_data['train_losses'], val_data=training_data['val_losses'], metric_name='Loss',
        title=f'ViT Training vs Validation Loss for {params.title}', save_path=f'{folder_path}/figures/loss.png')

    # Optionally, plot Training Accuracy vs Validation Accuracy
    plot_metrics(train_data=training_data['train_accuracies'], val_data=training_data['val_accuracies'], metric_name='Accuracy',
                 title=f'ViT Training vs Validation Loss for {params.title}',save_path=f'{folder_path}/figures/accuracy.png')

    print(f"\nFinetuning process complete with final test accuracy: {training_data['final_test_accuracy']:.2f}%")
    if params.save_model:
        torch.save(final_model_checkpoint, os.path.join(folder_path, 'final_model_checkpoint.pth'))

    with open(f"{folder_path}/results.json", "w") as f:
        json.dump(training_data, f, indent=4)

    return training_data

# def finetune(params, device):
#     # Setup and Data Preparation
#
#     if params.bottleneck_path is not None:
#         model = prepare_bottleneck_model(params.num_classes, params.bottleneck_dim, params.bottleneck_path, device, patch_size=params.patch_size)
#
#         if params.freeze_body: model.freeze_except_bottleneck()
#
#         for param in model.heads.parameters():
#             param.requires_grad = not params.freeze_head
#     else:
#         model = prepare_original_model(params.num_classes, device, patch_size=params.patch_size)
#
#         for param in model.parameters():
#             param.requires_grad = not params.freeze_body
#
#         for param in model.heads.parameters():
#             param.requires_grad = not params.freeze_head
#
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.AdamW(model.parameters(), lr=params.bottleneck_finetune_lr, weight_decay=params.weight_decay)
#     scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs, eta_min=params.min_anneal)
#
#     return train_and_plot(model, criterion, optimizer, scheduler, params, device)


def finetune_unfrozen(params, device, parallel=False):
    slow_lr = params.body_finetune_lr
    fast_lr = params.bottleneck_finetune_lr
    min_anneal = params.min_anneal

    print("Running experiment: ", params.title)
    print("Description: ", params.desc)

    if params.bottleneck_path is not None:
        print("Using bottleneck model")
        model = prepare_bottleneck_model(params.num_classes, params.bottleneck_dim, params.bottleneck_path, device,
                                         patch_size=params.patch_size, parallel=parallel)

        if params.freeze_body: model.freeze_except_bottleneck()

        # Freeze position embedding if it exists
        if hasattr(model, 'pos_embedding'):
            model.pos_embedding.requires_grad = not params.freeze_embeddings

        # Freeze class token if it exists
        if hasattr(model, 'class_token'):
            model.class_token.requires_grad = not params.freeze_body

        for param in model.conv_proj.parameters():
            param.requires_grad = not params.freeze_embeddings

        for param in model.heads.parameters():
            param.requires_grad = not params.freeze_head

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW([
            {'params': model.conv_proj.parameters(), 'lr': slow_lr},
            {'params': model.encoder.parameters(), 'lr': slow_lr},
            {'params': model.bottleneck.parameters(), 'lr': fast_lr},
            {'params': model.heads.parameters(), 'lr': fast_lr}
        ], weight_decay=params.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs, eta_min=min_anneal)
    else:
        print("Using original model")
        model = prepare_original_model(params.num_classes, device, patch_size=params.patch_size, parallel=parallel)


        for param in model.conv_proj.parameters():
            param.requires_grad = not params.freeze_embeddings

        for param in model.encoder.parameters():
            param.requires_grad = not params.freeze_body

        for param in model.heads.parameters():
            param.requires_grad = not params.freeze_head

        # Freeze position embedding if it exists
        if hasattr(model, 'pos_embedding'):
            model.pos_embedding.requires_grad = not params.freeze_embeddings

        # Freeze class token if it exists
        if hasattr(model, 'class_token'):
            model.class_token.requires_grad = not params.freeze_body

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW([
            {'params': model.conv_proj.parameters(), 'lr': slow_lr},
            {'params': model.encoder.parameters(), 'lr': slow_lr},
            {'params': model.heads.parameters(), 'lr': fast_lr}
        ], weight_decay=params.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs, eta_min=min_anneal)

    return train_and_plot(model, criterion, optimizer, scheduler, params, device)




def split_dataset(dataset, val_fraction=0.2):
# Get train indices for the split
#     train_idx, val_idx = train_test_split(
#         range(len(dataset)),
#         test_size=val_fraction,
#         random_state=42,
#         shuffle=False
#     )
#
#     # Create subset datasets
#     train_dataset = Subset(dataset, train_idx)
#     pretrain_dataset = Subset(dataset, val_idx)

    train_size = int((1 - val_fraction) * len(dataset))  # 90% train
    val_size = len(dataset) - train_size   # 10% val

    # Deterministic split
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    return train_dataset, val_dataset


def plot_metrics(train_data, val_data, metric_name, title, save_path=None, show=True):
    """
    General function to plot training and validation metrics over epochs.

    Args:
        train_data (list): List of training metric values (e.g., loss or accuracy).
        val_data (list): List of validation metric values.
        metric_name (str): Name of the metric (e.g., 'Loss', 'Accuracy').
        title (str): Title of the plot.
        save_path (str, optional): Path to save the plot. If None, it just displays.
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_data) + 1)

    # Plot Training Data
    plt.plot(epochs, train_data, 'b-o', label=f'Training {metric_name}', linewidth=2)
    # Plot Validation Data
    plt.plot(epochs, val_data, 'r-o', label=f'Validation {metric_name}', linewidth=2)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(epochs)  # Ensure all epochs are shown on the x-axis

    if save_path:
        plt.savefig(save_path)
        print(f"\nPlot saved to {save_path}")
    if show:
        plt.show()





seed = 42

def experiment_compression_vs_accuracy_img_net(device):
    # BASELINE
    retrieval_params = Experiment(
        title="TinyImageNet",
        desc="retrieve activations from imagenet",
        bottleneck_path=None,
        patch_size=32,
        batch_size=64,
        num_classes=200,
        dataset="TinyImageNet"
    )

    # retrieve_activations(retrieval_params, device)
    #
    # train_bottleneck_unsupervised("TinyImageNet_bottleneck_unsupervised_P32", "activations_TinyImageNet", device, bottleneck_dim=48, epochs=50)
    # train_bottleneck_unsupervised("TinyImageNet_bottleneck_unsupervised_P32", "activations_TinyImageNet", device, bottleneck_dim=96, epochs=50)
    # train_bottleneck_unsupervised("TinyImageNet_bottleneck_unsupervised_P32", "activations_TinyImageNet", device, bottleneck_dim=192, epochs=50)
    # train_bottleneck_unsupervised("TinyImageNet_bottleneck_unsupervised_P32", "activations_TinyImageNet", device, bottleneck_dim=384, epochs=50)

    # BASELINE
    experiment_params = Experiment(
        title="TinyImageNet_baseline",
        desc="baseline on tiny_image_net with 5 epochs",
        bottleneck_path=None,
        patch_size=32,
        batch_size=384,
        epochs=5,
        lr=1e-4,
        freeze_body=False,
        freeze_head=False,
        pre_train=False,
        dataset="TinyImageNet",
        num_classes=200,
    )

    # finetune_unfrozen(experiment_params, device)

    experiment_params = Experiment(
        title="TinyImageNet_bottleneck_2x",
        desc="base setup with 2x",
        bottleneck_path="models/TinyImageNet_bottleneck_unsupervised_P32_384.pth",
        patch_size=32,
        bottleneck_dim=384,
        batch_size=384,
        epochs=5,
        lr=1e-3,  # not used
        freeze_body=False,
        freeze_head=False,
        freeze_embeddings=False,
        dataset="TinyImageNet",
        num_classes=200,
        pre_train=False,
    )

    # finetune_unfrozen(experiment_params, device)

    experiment_params = Experiment(
        title="TinyImageNet_bottleneck_4x",
        desc="base setup with 4x",
        bottleneck_path="models/TinyImageNet_bottleneck_unsupervised_P32_192.pth",
        patch_size=32,
        bottleneck_dim=192,
        batch_size=384,
        epochs=5,
        lr=1e-3,  # not used
        freeze_body=False,
        freeze_head=False,
        freeze_embeddings=False,
        dataset="TinyImageNet",
        num_classes=200,
        pre_train=False,
    )

    # finetune_unfrozen(experiment_params, device)

    experiment_params = Experiment(
        title="TinyImageNet_bottleneck_8x",
        desc="base setup with 8x",
        bottleneck_path="models/TinyImageNet_bottleneck_unsupervised_P32_96.pth",
        patch_size=32,
        bottleneck_dim=96,
        batch_size=384,
        epochs=5,
        lr=1e-3,  # not used
        freeze_body=False,
        freeze_head=False,
        freeze_embeddings=False,
        dataset="TinyImageNet",
        num_classes=200,
        pre_train=False,
    )

    # finetune_unfrozen(experiment_params, device)

    experiment_params = Experiment(
        title="TinyImageNet_bottleneck_16x",
        desc="base setup with 16x",
        bottleneck_path="models/TinyImageNet_bottleneck_unsupervised_P32_48.pth",
        patch_size=32,
        bottleneck_dim=48,
        batch_size=384,
        epochs=5,
        lr=1e-3,  # not used
        freeze_body=False,
        freeze_head=False,
        freeze_embeddings=False,
        dataset="TinyImageNet",
        num_classes=200,
        pre_train=False,
    )

    finetune_unfrozen(experiment_params, device)


def experiment_compression_vs_accuracy(dataset, device):
    # BASELINE
    experiment_params = Experiment(
        title="cifar10_baseline",
        desc="baseline on cifar10 with 5 epochs",
        bottleneck_path=None,
        patch_size=32,
        batch_size=128,
        epochs=10,
        lr=1e-4,
        freeze_body=False,
        freeze_head=False,
        pre_train=False,
        dataset="CIFAR10",
        num_classes=10,
    )

    finetune_unfrozen(experiment_params, device)

    experiment_params = Experiment(
        title="cifar10_bottleneck_2x",
        desc="base setup with 2x",
        bottleneck_path="models/cifar10_bottleneck_unsupervised_P32_384.pth",
        patch_size=32,
        bottleneck_dim=384,
        batch_size=128,
        epochs=10,
        lr=1e-3,  # not used
        freeze_body=False,
        freeze_head=False,
        freeze_embeddings=False,
        dataset="CIFAR10",
        num_classes=10,
        pre_train=False,
    )

    finetune_unfrozen(experiment_params, device)

    experiment_params = Experiment(
        title="cifar10_bottleneck_4x",
        desc="base setup with 4x",
        bottleneck_path="models/cifar10_bottleneck_unsupervised_P32_192.pth",
        patch_size=32,
        bottleneck_dim=192,
        batch_size=128,
        epochs=10,
        lr=1e-3,  # not used
        freeze_body=False,
        freeze_head=False,
        freeze_embeddings=False,
        dataset="CIFAR10",
        num_classes=10,
        pre_train=False,
    )

    finetune_unfrozen(experiment_params, device)

    experiment_params = Experiment(
        title="cifar10_bottleneck_8x",
        desc="base setup with 8x",
        bottleneck_path="models/cifar10_bottleneck_unsupervised_P32_96.pth",
        patch_size=32,
        bottleneck_dim=96,
        batch_size=128,
        epochs=10,
        lr=1e-3,  # not used
        freeze_body=False,
        freeze_head=False,
        freeze_embeddings=False,
        dataset="CIFAR10",
        num_classes=10,
        pre_train=False,
    )

    finetune_unfrozen(experiment_params, device)

    experiment_params = Experiment(
        title="cifar10_bottleneck_16x",
        desc="base setup with 16x",
        bottleneck_path="models/cifar10_bottleneck_unsupervised_P32_48.pth",
        patch_size=32,
        bottleneck_dim=48,
        batch_size=128,
        epochs=10,
        lr=1e-3,  # not used
        freeze_body=False,
        freeze_head=False,
        freeze_embeddings=False,
        dataset="CIFAR10",
        num_classes=10,
        pre_train=False,
    )

    # finetune_unfrozen(experiment_params, device)



if __name__ == '__main__':
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")




    torch.manual_seed(seed)

    from unsupervised import train_bottleneck_unsupervised, retrieve_activations

    # retrieve_activations(device)
    #
    # train_bottleneck_unsupervised("cifar10_bottleneck_unsupervised_P32", "activations_cifar10", device, bottleneck_dim=48, epochs=50)
    # train_bottleneck_unsupervised("cifar10_bottleneck_unsupervised_P32", "activations_cifar10", device, bottleneck_dim=96, epochs=50)
    # train_bottleneck_unsupervised("cifar10_bottleneck_unsupervised_P32", "activations_cifar10", device, bottleneck_dim=192, epochs=50)
    # train_bottleneck_unsupervised("cifar10_bottleneck_unsupervised_P32", "activations_cifar10", device, bottleneck_dim=384, epochs=50)
    # train_bottleneck_unsupervised("bottleneck_unsupervised_P16", "activations10k_16.pt", device, bottleneck_dim=96, epochs=100)
    # train_bottleneck_unsupervised("bottleneck_unsupervised_P16", "activations10k_16.pt", device, bottleneck_dim=48, epochs=100)

    # finetune_bottleneck("models/bottleneck_unsupervised_P32_96.pth", name="bottleneck_P32_96_finetune_heads",
    #                     bottleneck_dim=96, device=device, patch_size=32,
    #                     lr=1e-3, batch_size=64, epochs=10, num_classes=100)


    # experiment_compression_vs_accuracy("CIFAR10", device)
    experiment_compression_vs_accuracy_img_net(device)

    experiment_params = Experiment(
        title="bottleneck_16x",
        desc="base setup with 16x",
        bottleneck_path="models/bottleneck_unsupervised_P32_48.pth",
        patch_size=32,
        bottleneck_dim=48,
        batch_size=128,
        epochs=10,
        lr=1e-3,  # not used
        freeze_body=False,
        freeze_head=False,
        freeze_embeddings=False,
        pre_train=False,
    )

    # finetune_unfrozen(experiment_params, device)

    # BASELINE
    experiment_params = Experiment(
        title="cifar10_baseline",
        desc="baseline with 5 epochs",
        bottleneck_path=None,
        patch_size=32,
        batch_size=128,
        epochs=5,
        lr=1e-4,
        freeze_body=False,
        freeze_head=False,
        pre_train=False,
        dataset="CIFAR10",
        num_classes=10,
    )

    # finetune_unfrozen(experiment_params, device)

    experiment_params = Experiment(
        title="cifar10_bottleneck_2x",
        desc="base setup with 2x",
        bottleneck_path="models/bottleneck_unsupervised_P32_384.pth",
        patch_size=32,
        bottleneck_dim=384,
        batch_size=128,
        epochs=5,
        lr=1e-3,  # not used
        freeze_body=False,
        freeze_head=False,
        freeze_embeddings=False,
        pre_train=False,
    )

    # finetune_unfrozen(experiment_params, device, parallel=True)

    # params = {
    #     experiment_name: "bottleneck_P32_96_finetune_heads"
    # }

    #
    # finetune_bottleneck("models/bottleneck_unsupervised_P16_96.pth", name="bottleneckv2_P16_192_finetune_heads", bottleneck_dim=192,
    #                     device=device,
    #                     lr=5e-3, batch_size=64, epochs=10, num_classes=100)
    #
    # finetune_bottleneck("models/bottleneck_unsupervised_P16_48.pth", name="bottleneckv2_P16_192_finetune_heads", bottleneck_dim=192,
    #                     device=device,
    #                     lr=5e-3, batch_size=64, epochs=10, num_classes=100)
    #
    # finetune_original(device, name="original_P32_finetune_heads2", lr=1e-3, batch_size=64, epochs=10, num_classes=100)