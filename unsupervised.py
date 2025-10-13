

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from params import Experiment

from tqdm import tqdm


from bottleneck import Bottleneck
from bottleneck_vision_transformer import BottleneckVisionTransformer
import matplotlib.pyplot as plt
from main import prepare_dataset, prepare_original_model, split_dataset
from torchdistill.core.forward_hook import ForwardHookManager

def retrieve_activations(params, device):
    # train_loader, test_loader, pretrain_loader, pretrain_val_loader =  prepare_dataset(64)


    _, _, _, pretrain_loader, pretrain_val_loader =  prepare_dataset(params)

    model = prepare_original_model(params.num_classes, device, patch_size=params.patch_size)

    model.eval()

    forward_hook_manager = ForwardHookManager(device)
    forward_hook_manager.add_hook(model, 'conv_proj', requires_input=True, requires_output=True)

    all_activations = []

    with torch.no_grad():
        for inputs, _ in tqdm(pretrain_loader, desc='Evaluating'):
            inputs = inputs.to(device)

            # Forward pass (hook stores activations)
            _ = model(inputs)

            io_dict = forward_hook_manager.pop_io_dict()
            conv_out = io_dict['conv_proj']['output']  # shape: (B, hidden_dim, h, w)

            B, hidden_dim, h, w = conv_out.shape
            n_patches = h * w

            # reshape to (B, n_patches, hidden_dim)
            conv_out = conv_out.reshape(B, hidden_dim, n_patches).permute(0, 2, 1)

            all_activations.append(conv_out.cpu())

    with torch.no_grad():
        for inputs, _ in tqdm(pretrain_val_loader, desc='Evaluating Val Loader'):
            inputs = inputs.to(device)

            # Forward pass (hook stores activations)
            _ = model(inputs)

            io_dict = forward_hook_manager.pop_io_dict()
            conv_out = io_dict['conv_proj']['output']  # shape: (B, hidden_dim, h, w)

            B, hidden_dim, h, w = conv_out.shape
            n_patches = h * w

            # reshape to (B, n_patches, hidden_dim)
            conv_out = conv_out.reshape(B, hidden_dim, n_patches).permute(0, 2, 1)

            all_activations.append(conv_out.cpu())

    # Concatenate all batches -> (N, n_patches, hidden_dim)
    all_activations = torch.cat(all_activations, dim=0)
    # Merge N and n_patches dimensions -> (N * n_patches, hidden_dim)
    # all_activations = all_activations.reshape(-1, all_activations.shape[-1])


    print("All activation shape", all_activations.shape)
    torch.save(all_activations.cpu(), f'processed_data/activations_{params.title}.pt')
    return all_activations  # (10000, 49, 768) for P=32


class ActivationsDataset(Dataset):
    def __init__(self, activations):
        self.activations = activations # Convert numpy to tensor if needed

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx]



def train_bottleneck_unsupervised(name, activations_path, device, epochs = 50, bottleneck_dim = 384):

    activations = torch.load(f'processed_data/{activations_path}.pt')
    # print(activations.shape)
    # print(activations)
    dataset = ActivationsDataset(activations)
    train_dataset, val_dataset = split_dataset(dataset, val_fraction=0.1)

    # split into train and val

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    model = Bottleneck(768, bottleneck_dim)
    criterion = nn.MSELoss()
    # criterion = nn.CosineEmbeddingLoss()
    # optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=5e-5)
    train_losses = []
    val_losses = []

    model.to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total number of parameters: {total_params}')

    # Training loop
    print("\nStarting training...\n")
    best_loss = float('inf')

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}]")

        train_loss, losses_epoch = train_epoch_bottleneck(model, train_dataloader, criterion,
                                            optimizer, device)
        val_loss = evaluate_bottleneck(model, val_dataloader, criterion, device)

        scheduler.step()
        train_losses.append(train_loss)
        val_losses.append(val_loss)


        # losses.extend(losses_epoch)

        print(f"Train Loss: {train_loss:.4f}")
        # print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\n")

        # Save best model (this overfits)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, f'models/{name}_{bottleneck_dim}.pth')
            print(f"Saved best model with val loss: {val_loss:.6f}\n")

    print(f"\nTraining completed! Best val loss: {best_loss:.6f}")
    # use log scale

    plt.style.use('fivethirtyeight')
    plt.figure(figsize=(10, 6))
    plt.semilogy(train_losses, label='Training Loss', linewidth=2)  # Use log scale on y-axis
    plt.semilogy(val_losses, label='Validation Loss', linewidth=2)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.xlabel('Iterations')
    plt.ylabel('Loss (log scale)')
    plt.title('Unsupervised Bottleneck Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'figures/unsupervised_bottleneck_training_loss_{bottleneck_dim}.png')
    plt.show()


    # # Load and evaluate best model
    # print("\nLoading best model for final evaluation...")
    # checkpoint = torch.load('best_model.pth')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # final_loss, final_acc = evaluate(model, test_loader, criterion, device)
    # print(f"Final Test Accuracy: {final_acc:.2f}%")


# Bottleneck usual input (B x Sequence length x Hidden dim), (B x 49 x 768)
#
#
#


def train_epoch_bottleneck(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    losses = []

    pbar = tqdm(loader, desc='Training')
    for inputs in pbar:
        inputs = inputs.to(device)

        optimizer.zero_grad()

        reconstructed = model(inputs)
        loss = criterion(reconstructed, inputs)


        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        running_loss += loss.item()


        pbar.set_postfix({'loss': running_loss / len(pbar)})

    return running_loss / len(loader), losses

# Evaluation function
def evaluate_bottleneck(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs in tqdm(loader, desc='Evaluating'):
            inputs = inputs.to(device)
            reconstructed = model(inputs)
            loss = criterion(reconstructed, inputs)

            running_loss += loss.item()

    return running_loss / len(loader)