import json
import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from params import Experiment
from plots import plot_metrics
from prepare import prepare_original_model, prepare_dataset, prepare_bottleneck_model


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



def finetune(params, device, parallel=False):
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


seed = 42

if __name__ == '__main__':
    # Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    torch.manual_seed(seed)

    from unsupervised import train_bottleneck_unsupervised, retrieve_activations

    # retrieve_activations(device)

    # train_bottleneck_unsupervised("cifar10_bottleneck_unsupervised_P32", "activations_cifar10", device, bottleneck_dim=48, epochs=50)
    # train_bottleneck_unsupervised("cifar10_bottleneck_unsupervised_P32", "activations_cifar10", device, bottleneck_dim=96, epochs=50)
    # train_bottleneck_unsupervised("cifar10_bottleneck_unsupervised_P32", "activations_cifar10", device, bottleneck_dim=192, epochs=50)
    # train_bottleneck_unsupervised("cifar10_bottleneck_unsupervised_P32", "activations_cifar10", device, bottleneck_dim=384, epochs=50)

    retrieval_params = Experiment(
        title="TinyImageNet_train_all",
        desc="retrieve activations from imagenet",
        bottleneck_path=None,
        patch_size=32,
        batch_size=64,
        num_classes=200,
        dataset="TinyImageNet",
    )

    # retrieve_activations(retrieval_params, device)

    from experiments import pre_train_bottleneck, transfer_bottleneck

    # pre_train_bottleneck("TinyImageNet", [192], [0.1, 0.05, 0.02, 0.01], device)

    transfer_bottleneck("TinyImageNet", "CIFAR100", [192], [0.1, 0.05, 0.02, 0.01], device)
    from experiments import experiment_compression_vs_accuracy_general, pre_train_bottleneck

    # experiment_compression_vs_accuracy_general("CalTech256", device)



 