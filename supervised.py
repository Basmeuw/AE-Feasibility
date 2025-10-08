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