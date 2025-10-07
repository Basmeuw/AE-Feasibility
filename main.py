import copy

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.models import vit_b_32, ViT_B_32_Weights, VisionTransformer

from tqdm import tqdm
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np

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


def train( model, train_loader, test_loader, criterion, optimizer, scheduler, device, NUM_EPOCHS):
    # Training loop
    print("\nStarting training...\n")
    best_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                            optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, 'model_new_head.pth')
            print(f"✓ Saved best model with accuracy: {best_acc:.2f}%\n")

    print(f"\nTraining completed! Best test accuracy: {best_acc:.2f}%")

    # Load and evaluate best model
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load('model_new_head.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Accuracy: {final_acc:.2f}%")

def prepare_dataset(BATCH_SIZE):
    # Data augmentation and normalization
    train_transform = transforms.Compose([
        transforms.Resize(224),
        # transforms.RandomCrop(224, padding=4),
        # transforms.RandomHorizontalFlip(),
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
                                           download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False,
                                     download=True, transform=test_transform)

    # Get train indices for the split
    train_idx, val_idx = train_test_split(
        range(len(train_dataset_full)),
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # Create subset datasets
    train_dataset = Subset(train_dataset_full, train_idx)
    pretrain_dataset_full = Subset(train_dataset_full, val_idx)

    pretrain_dataset, pretrain_val_dataset = split_dataset(pretrain_dataset_full, val_fraction=0.1)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=4, pin_memory=True)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=BATCH_SIZE)
    pretrain_val_loader = DataLoader(pretrain_val_dataset, batch_size=BATCH_SIZE)

    return train_loader, test_loader, pretrain_loader, pretrain_val_loader

def prepare_original_model(device):
    # Load pre-trained Vision Transformer B/32 (4x faster than B/16)
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

    return original_model

def prepare_bottleneck_model(bottleneck_dim, path, device):
    # 1. Recreate architecture
    bottleneck = Bottleneck(embedding_dim=768, bottleneck_dim=bottleneck_dim)

    # 2. Load pretrained weights
    saved_model = torch.load(path, map_location=device)

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

    return bottleneck_model

def prepare_models(NUM_CLASSES, device):

    original_model = prepare_original_model(device)

    bottleneck_model = prepare_bottleneck_model(768, "models/bottleneck_best_model_384.pth", device)
    # 1. Create ONE new classification head
    new_head = nn.Linear(original_model.heads[0].in_features, NUM_CLASSES)

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

    train_loader, test_loader, pretrain_loader, pretrain_val_loader = prepare_dataset(BATCH_SIZE)

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




def train_bottleneck_supervised(teacher, student: BottleneckVisionTransformer, pretrain_loader, pretrain_val_loader, device):
    pre_trained = torch.load("model_new_head.pth", map_location=device)

    # Extract the full state dict
    state_dict = pre_trained["model_state_dict"]

    # Filter out only the parameters belonging to the 'heads' layer
    head_state_dict = {k.replace("heads.", ""): v for k, v in state_dict.items() if k.startswith("heads.")}


    # Load these weights into each model’s head
    teacher.heads.load_state_dict(head_state_dict)
    student.heads.load_state_dict(head_state_dict)

    teacher.eval()
    student.use_bottleneck = True  # IMPORTANT: Activate the bottleneck
    student.freeze_except_bottleneck()

    NUM_EPOCHS = 20
    LR = 1e-3

    optimizer = optim.AdamW(student.bottleneck.parameters(), lr=LR)  # Optimize only bottleneck
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


    print("\nStarting supervised bottleneck training...\n")
    best_acc = 0.0

    # Store history for plotting
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    val_losses = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}]")

        # Get loss and accuracy from training
        train_loss, _, train_acc = train_epoch_bottleneck_supervised(student, teacher, pretrain_loader, optimizer,
                                                                     device)

        # Get loss and accuracy from evaluation
        val_loss, val_acc = evaluate_bottleneck_supervised(student, teacher, pretrain_val_loader, device)

        val_losses.append(val_loss)

        scheduler.step()

        # Log history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f}, Val Acc:   {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")

        # Save the model if it has the best validation accuracy so far
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, f'models/bottleneck_supervised_best_model_{student.bottleneck.bottleneck_dim}.pth')
            print(f"New best model saved with validation accuracy: {val_acc:.2f}%\n")

    print(f"\nTraining completed! Best validation accuracy: {best_acc:.2f}%")
    # You can now plot the history dict

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 6))
    plt.semilogy(val_losses, label='Validation Loss', linewidth=2)  # Use log scale on y-axis
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Iterations')
    plt.ylabel('Loss (log scale)')
    plt.title('Supervised Bottleneck Validation Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/supervised_bottleneck_validation_loss.png')
    plt.show()




def train_epoch_bottleneck_supervised(student, teacher, loader, optimizer, device):
    student.eval()  # Set student to training mode
    teacher.eval()  # Ensure teacher is in evaluation mode

    running_loss = 0.0
    total_correct = 0
    total_samples = 0
    losses_epoch = []

    criterion_soft = nn.KLDivLoss(reduction='batchmean')
    T = 2.0  # Temperature for soft distillation

    pbar = tqdm(loader, desc='Training Epoch')
    for inputs, _ in pbar:
        inputs = inputs.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Get predictions
        with torch.no_grad():
            teacher_predictions = teacher(inputs)
        student_predictions = student(inputs)

        # Calculate soft distillation loss
        loss = criterion_soft(
            F.log_softmax(student_predictions / T, dim=1),
            F.softmax(teacher_predictions / T, dim=1)
        ) * (T * T)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        losses_epoch.append(loss.item())

        # No gradients needed for accuracy calculation
        with torch.no_grad():
            teacher_labels = torch.argmax(teacher_predictions, dim=1)
            student_labels = torch.argmax(student_predictions, dim=1)
            total_correct += (student_labels == teacher_labels).sum().item()
            total_samples += inputs.size(0)

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{(total_correct / total_samples) * 100:.2f}%'
        })

    epoch_loss = running_loss / len(loader)
    epoch_acc = (total_correct / total_samples) * 100
    return epoch_loss, losses_epoch, epoch_acc


def evaluate_bottleneck_supervised(student, teacher, loader, device):
    student.eval()  # Set student to evaluation mode
    teacher.eval()

    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    criterion_hard = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc='Evaluating'):
            inputs = inputs.to(device)

            # Get predictions
            teacher_predictions = teacher(inputs)
            student_predictions = student(inputs)

            # Get hard labels from teacher for loss and accuracy
            teacher_labels = torch.argmax(teacher_predictions, dim=1)

            # Calculate hard loss
            loss = criterion_hard(student_predictions, teacher_labels)
            running_loss += loss.item()

            # Calculate accuracy
            student_labels = torch.argmax(student_predictions, dim=1)
            total_correct += (student_labels == teacher_labels).sum().item()
            total_samples += inputs.size(0)

    val_loss = running_loss / len(loader)
    val_acc = (total_correct / total_samples) * 100
    return val_loss, val_acc

def retrieve_predictions(pretrain_loader, model, device):


    model.eval()

    all_predictions = []

    with torch.no_grad():
        for inputs, _ in tqdm(pretrain_loader, desc='Evaluating'):
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)  # shape: (B, num_classes)

            all_predictions.append(outputs.cpu())

    # Concatenate all batches -> (N, num_classes)
    all_predictions = torch.cat(all_predictions, dim=0)
    print(all_predictions.shape)
    torch.save(all_predictions.cpu(), 'processed_data/predictions10k.pt')
    return all_predictions  # (10000, 100)



def split_dataset(dataset, val_fraction=0.2):
# Get train indices for the split
    train_idx, val_idx = train_test_split(
        range(len(dataset)),
        test_size=val_fraction,
        random_state=42,
        shuffle=True
    )

    # Create subset datasets
    train_dataset = Subset(dataset, train_idx)
    pretrain_dataset = Subset(dataset, val_idx)

    return train_dataset, pretrain_dataset

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    from unsupervised import train_bottleneck_unsupervised, retrieve_activations
    # retrieve_activations(device)
    # main()
    train_bottleneck_unsupervised("bottleneck_unsupervised", device, epochs=100)