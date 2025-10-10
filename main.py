import copy
import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
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
            }, 'model_new_head.pth')
            print(f"✓ Saved best model with loss: {val_loss:.4f}, acc: {val_acc:.2f}%\n")

            # plot_metrics(train_losses, val_losses, , val_accuracies)

    print(f"\nTraining completed! Best validation loss: {best_loss:.4f} , acc: {best_acc:.2f}%")

    # Load and evaluate best model
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load('model_new_head.pth')
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
    }



def prepare_dataset(BATCH_SIZE):
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

    # Load CIFAR-100 dataset
    print("Loading CIFAR-100 dataset...")
    train_dataset_full = datasets.CIFAR100(root='./data', train=True,
                                           download=True)
    test_dataset = datasets.CIFAR100(root='./data', train=False,
                                     download=True, transform=test_transform)

    train_dataset2, val_dataset = split_dataset(train_dataset_full, 0.1)

    train_dataset, pretrain_dataset = split_dataset(train_dataset2, 0.2)

    pretrain_dataset, pretrain_val_dataset = split_dataset(pretrain_dataset, val_fraction=0.1)

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_transform
    pretrain_dataset.dataset.transform = train_transform
    pretrain_val_dataset.dataset.transform = test_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False )
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE, shuffle=True)
    pretrain_val_loader = DataLoader(pretrain_val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, val_loader, test_loader, pretrain_loader, pretrain_val_loader

def prepare_original_model(num_classes, device, patch_size=16):
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

    original_model.to(device)

    return original_model

def prepare_bottleneck_model(num_classes, bottleneck_dim, path, device, patch_size=16):
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
    bottleneck_model = bottleneck_model.to(device)

    return bottleneck_model

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

def main():


    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 10
    NUM_CLASSES = 100

    train_loader, val_loader, test_loader, pretrain_loader, pretrain_val_loader = prepare_dataset(BATCH_SIZE)

    original_model, bottleneck_model = prepare_models(NUM_CLASSES, device)

    # bottleneck_model.heads = nn.Linear(original_model.heads[0].in_features, NUM_CLASSES)


    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(original_model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # train(original_model, pretrain_loader, pretrain_val_loader, criterion, optimizer, scheduler, device, NUM_EPOCHS)

    # train_bottleneck_unsupervised(device, bottleneck_dim=384)
    # train_bottleneck_supervised(original_model, bottleneck_model, pretrain_loader, pretrain_val_loader, device)
    # retrieve_predictions(pretrain_loader, model , device)


# wrapper function that takes care of storing the data
def train_and_plot(model, criterion, optimizer, scheduler, params, device):

    # Convert Params instance to dict
    params_dict = params.__dict__


    train_loader, val_loader, test_loader, pretrain_loader, pretrain_val_loader = prepare_dataset(params.batch_size)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    folder_path = 'runs/run_{}_{}'.format(timestamp, params.title)
    os.makedirs(folder_path, exist_ok=True)
    os.makedirs(os.path.join(folder_path, 'figures'), exist_ok=True)
    figure_path = os.path.join(folder_path, 'figures')

    # Dump to JSON file
    with open(folder_path + '/params.json', 'w') as f:
        json.dump(params_dict, f, indent=4)

    # if params.pretrained:
    #     pass

    # Training
    training_data = train(
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

    # Plot Training Loss vs Validation Loss
    plot_metrics(train_data=training_data['train_losses'], val_data=training_data['val_losses'], metric_name='Loss',
        title=f'ViT Training vs Validation Loss for {params.title}', save_path=f'{folder_path}/figures/loss.png')

    # Optionally, plot Training Accuracy vs Validation Accuracy
    plot_metrics(train_data=training_data['train_accuracies'], val_data=training_data['val_accuracies'], metric_name='Accuracy',
                 title=f'ViT Training vs Validation Loss for {params.title}',save_path=f'{folder_path}/figures/accuracy.png')

    print(f"\nFinetuning process complete with final test accuracy: {training_data['final_test_accuracy']:.2f}%")

    with open(f"{folder_path}/results.json", "w") as f:
        json.dump(training_data, f, indent=4)

    return training_data

def finetune(params, device):
    # Setup and Data Preparation

    if params.bottleneck_path is not None:
        model = prepare_bottleneck_model(params.num_classes, params.bottleneck_dim, params.bottleneck_path, device, patch_size=params.patch_size)

        if params.freeze_body: model.freeze_except_bottleneck()

        for param in model.heads.parameters():
            param.requires_grad = not params.freeze_head
    else:
        model = prepare_original_model(params.num_classes, device, patch_size=params.patch_size)

        for param in model.parameters():
            param.requires_grad = not params.freeze_body

        for param in model.heads.parameters():
            param.requires_grad = not params.freeze_head

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=params.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs, eta_min=3e-5)

    return train_and_plot(model, criterion, optimizer, scheduler, params, device)


def finetune_unfrozen(params, device):
    slow_lr = 1e-4
    fast_lr = 2e-3
    min_anneal = 5e-5

    if params.bottleneck_path is not None:
        model = prepare_bottleneck_model(params.num_classes, params.bottleneck_dim, params.bottleneck_path, device,
                                         patch_size=params.patch_size)

        if params.freeze_body: model.freeze_except_bottleneck()



        for param in model.heads.parameters():
            param.requires_grad = params.freeze_head

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW([
            {'params': model.conv_proj.parameters(), 'lr': slow_lr},
            {'params': model.encoder.parameters(), 'lr': slow_lr},
            {'params': model.bottleneck.parameters(), 'lr': fast_lr},
            {'params': model.heads.parameters(), 'lr': fast_lr}
        ], weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=params.epochs, eta_min=min_anneal)
    else:
        model = prepare_original_model(params.num_classes, device, patch_size=params.patch_size)

        for param in model.parameters():
            param.requires_grad = False

        for param in model.heads.parameters():
            param.requires_grad = params.freeze_head

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW([
            {'params': model.conv_proj.parameters(), 'lr': slow_lr},
            {'params': model.encoder.parameters(), 'lr': slow_lr},
            {'params': model.heads.parameters(), 'lr': fast_lr}
        ], weight_decay=0.01)
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

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")




    torch.manual_seed(seed)

    from unsupervised import train_bottleneck_unsupervised, retrieve_activations
    # retrieve_activations(device)
    # main()
    # train_bottleneck_unsupervised("bottleneck_unsupervised_P32", "activations10k", device, bottleneck_dim=192, epochs=50)
    # train_bottleneck_unsupervised("bottleneck_unsupervised_P16", "activations10k_16.pt", device, bottleneck_dim=96, epochs=100)
    # train_bottleneck_unsupervised("bottleneck_unsupervised_P16", "activations10k_16.pt", device, bottleneck_dim=48, epochs=100)

    # finetune_bottleneck("models/bottleneck_unsupervised_P32_96.pth", name="bottleneck_P32_96_finetune_heads",
    #                     bottleneck_dim=96, device=device, patch_size=32,
    #                     lr=1e-3, batch_size=64, epochs=10, num_classes=100)

    experiment_params = Experiment(
        title="bottleneck_P32_48_finetune_heads",
        bottleneck_path="models/bottleneck_unsupervised_P32_48.pth",
        patch_size=32,
        bottleneck_dim=48,
        num_classes=100,
        embed_dim=768,
        batch_size=64,
        epochs=5,
        lr=1e-3,
        freeze_body=True,
        freeze_head=False,
    )

    # finetune(experiment_params, device)

    experiment_params = Experiment(
        title="pretrain_unfrozen_bottleneck_4x",
        desc="with 1 pretrain epoch, also with cosine anneal now",
        bottleneck_path="models/bottleneck_unsupervised_P32_192.pth",
        patch_size=32,
        bottleneck_dim=192,
        embed_dim=768,
        batch_size=64,
        epochs=10,
        lr=1e-3, # not used
        freeze_body=True,
        freeze_head=False,
    )

    # finetune_unfrozen(experiment_params, device)

    # BASELINE
    experiment_params = Experiment(
        title="baseline_P32_1e-4",
        desc="baseline for patch 32 No head pretraining",
        bottleneck_path=None,
        patch_size=32,
        bottleneck_dim=192,
        embed_dim=768,
        batch_size=64,
        epochs=10,
        lr=1e-4,  # not used
        freeze_body=False,
        freeze_head=False,
    )

    finetune(experiment_params, device)

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