import torch

from bottleneck import Bottleneck
from main import finetune
from params import Experiment
from unsupervised import retrieve_activations, train_bottleneck_unsupervised

def get_num_classes(dataset):
    num_classes = 0
    if dataset == "CalTech256":
        num_classes = 257
    elif dataset == "TinyImageNet":
        num_classes = 200
    elif dataset == "CIFAR10":
        num_classes = 10
    elif dataset == "CIFAR100":
        num_classes = 100
    elif dataset == "Food101":
        num_classes = 101
    else:
        print(f"Dataset {dataset} not supported.")
        return -1
    return num_classes

# Used for the scenario where we pre-train the bottleneck on a single dataset only
def pre_train_bottleneck(dataset, bottleneck_dims, data_fractions, device):
    num_classes = get_num_classes(dataset)

    retrieval_params = Experiment(
        title=dataset,
        desc="retrieve activations from imagenet",
        bottleneck_path=None,
        patch_size=32,
        batch_size=64,
        num_classes=num_classes,
        dataset=dataset
    )

    # bottleneck_dims = [384, 192, 96, 48]


    for bottleneck_dim in bottleneck_dims:
        for data_fraction in data_fractions:
            print(f"Pre-training, bottleneck dim {bottleneck_dim}, data_fraction {data_fraction}, for dataset {dataset}")
            train_bottleneck_unsupervised(data_fraction, f"{dataset}_{data_fraction}_bottleneck_unsupervised_P32", f"activations_{dataset}", device,
                                          bottleneck_dim=bottleneck_dim, epochs=50)

# Used for the scenario where we pre-train the bottleneck on a single dataset only. Here we finetune with the generated datasets
def transfer_bottleneck_data_fraction(pre_train_dataset, dataset, bottleneck_dims, data_fractions, device, save_folder):
    num_classes = get_num_classes(dataset)


    for bottleneck_dim in bottleneck_dims:
        for data_fraction in data_fractions:
            print(f"Fine-tuning, bottleneck dim {bottleneck_dim}, data_fraction {data_fraction}, for dataset {dataset}")
            bottleneck_ratio = 768 / bottleneck_dim
            experiment_params = Experiment(
                title=f"{dataset}_{data_fraction}_bottleneck_transfer_{bottleneck_ratio}x",
                desc=f"Testing scenario 1. Transferring bottleneck pre-trained on fraction {data_fraction} of {pre_train_dataset}, and now finetuning on {dataset}. Base setup with {bottleneck_ratio}x",
                bottleneck_path=f"models/{pre_train_dataset}_{data_fraction}_bottleneck_unsupervised_P32_{bottleneck_dim}.pth",
                patch_size=32,
                bottleneck_dim=bottleneck_dim,
                batch_size=384,
                epochs=1,
                lr=1e-3,  # not used
                freeze_body=False,
                freeze_head=False,
                freeze_embeddings=False,
                dataset=dataset,
                num_classes=num_classes,
                pre_train=False,
            )
            finetune(experiment_params, device, save_folder)


def transfer_bottleneck_data_fraction_full_self( device):

    bottleneck_dims = [192]
    data_fractions = [0.1, 0.01, 0.001]

    pre_train_dataset = "Food101"
    train_dataset = "Food101"

    retrieval_params = Experiment(
        title=f"{pre_train_dataset}",
        desc="retrieve activations",
        bottleneck_path=None,
        patch_size=32,
        batch_size=64,
        num_classes=get_num_classes(dataset="Food101"),
        dataset=pre_train_dataset,
    )

    retrieve_activations(retrieval_params, device)

    # First pre_train the bottleneck with the possible data fractions
    pre_train_bottleneck(pre_train_dataset, bottleneck_dims, data_fractions, device)

    # # Create a randomly initialized bottleneck
    # model = Bottleneck(768, 192)
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    # }, f'models/TinyImageNet_0_bottleneck_unsupervised_P32_192.pth') # HARDCODED

    experiment_folder = f"useful_runs/self_transfer_data_fraction_{train_dataset}"

    # experiment_compression_vs_accuracy_general("Food101", device, baseline_only=True, save_folder=experiment_folder)
    transfer_bottleneck_data_fraction(pre_train_dataset, train_dataset, bottleneck_dims, data_fractions, device, save_folder=experiment_folder)

    from plots import plot_metric_from_runs, plot_final_metric_bar_chart
    plot_metric_from_runs(
        parent_folder=experiment_folder,
        metric_name="val_losses",
        title="Food101 Self Transfer Val Losses",
        save_path="val_loss"
    )

    plot_metric_from_runs(
        parent_folder=experiment_folder,
        metric_name="val_accuracies",
        title="Food101 Self Transfer Val Accuracy",
        save_path="val_acc"
    )

    # 2. Plot the final test accuracy (Bar Chart)
    plot_final_metric_bar_chart(
        parent_folder=experiment_folder,
        final_metric_key="final_test_accuracy",
        title="Food101 Self Transfer Final Test acc",
        save_path="final_test"
    )

def transfer_bottleneck_data_fraction_full(device):
    bottleneck_dims = [192]
    data_fractions = [0.1, 0.01, 0.001, 0]

    pre_train_dataset = "TinyImageNet"
    train_dataset = "Food101"

    retrieval_params = Experiment(
        title=pre_train_dataset,
        desc="retrieve activations",
        bottleneck_path=None,
        patch_size=32,
        batch_size=64,
        num_classes=get_num_classes(dataset="Food101"),
        dataset=pre_train_dataset,
    )

    retrieve_activations(retrieval_params, device)

    # First pre_train the bottleneck with the possible data fractions
    pre_train_bottleneck(pre_train_dataset, bottleneck_dims, data_fractions, device)

    # Create a randomly initialized bottleneck
    model = Bottleneck(768, 192)
    torch.save({
        'model_state_dict': model.state_dict(),
    }, f'models/TinyImageNet_0_bottleneck_unsupervised_P32_192.pth') # HARDCODED

    experiment_folder = f"useful_runs/transfer_data_fraction_{train_dataset}_fixed_act"

    # experiment_compression_vs_accuracy_general("Food101", device, baseline_only=True, save_folder=experiment_folder)
    transfer_bottleneck_data_fraction(pre_train_dataset, train_dataset, bottleneck_dims, data_fractions, device,
                                      save_folder=experiment_folder)

    from plots import plot_metric_from_runs, plot_final_metric_bar_chart
    plot_metric_from_runs(
        parent_folder=experiment_folder,
        metric_name="val_losses",
        title="Food101 Transfer from TIN Val Losses",
        save_path="val_loss"
    )

    plot_metric_from_runs(
        parent_folder=experiment_folder,
        metric_name="val_accuracies",
        title="Food101 Transfer from TIN Val Accuracy",
        save_path="val_acc"
    )

    # 2. Plot the final test accuracy (Bar Chart)
    plot_final_metric_bar_chart(
        parent_folder=experiment_folder,
        final_metric_key="final_test_accuracy",
        title="Food101 Transfer from TIN Final Test acc",
        save_path="final_test"
    )

def experiment_compression_vs_accuracy_general(dataset, device, baseline_only=False, save_folder="runs"):
    num_classes = get_num_classes(dataset)
    epochs = 1
    retrieval_params = Experiment(
        title=dataset,
        desc="retrieve activations from imagenet",
        bottleneck_path=None,
        patch_size=32,
        batch_size=64,
        num_classes=num_classes,
        dataset=dataset
    )

    bottleneck_dims = [384, 192, 96, 48]
    bottleneck_ratios = [2, 4, 8, 16]

    #
    # retrieve_activations(retrieval_params, device)
    #
    # train_bottleneck_unsupervised(f"{dataset}_bottleneck_unsupervised_P32", f"activations_{dataset}", device, bottleneck_dim=48, epochs=50)
    # train_bottleneck_unsupervised(f"{dataset}_bottleneck_unsupervised_P32", f"activations_{dataset}", device, bottleneck_dim=96, epochs=50)
    # train_bottleneck_unsupervised(f"{dataset}_bottleneck_unsupervised_P32", f"activations_{dataset}", device, bottleneck_dim=192, epochs=50)
    # train_bottleneck_unsupervised(f"{dataset}_bottleneck_unsupervised_P32", f"activations_{dataset}", device, bottleneck_dim=384, epochs=50)

    # BASELINE
    experiment_params = Experiment(
        title=f"{dataset}_baseline",
        desc=f"baseline on {dataset}",
        bottleneck_path=None,
        patch_size=32,
        batch_size=384,
        epochs=epochs,
        lr=1e-4,
        freeze_body=False,
        freeze_head=False,
        pre_train=False,
        dataset=dataset,
        num_classes=num_classes,
    )

    finetune(experiment_params, device, save_folder)

    if baseline_only:
        return

    chosen_ratios = [0, 1, 2, 3] # sometimes we dont want to run all of them
    for i in chosen_ratios:
        bottleneck_dim = bottleneck_dims[i]
        bottleneck_ratio = bottleneck_ratios[i]
        experiment_params = Experiment(
            title=f"{dataset}_bottleneck_{bottleneck_ratio}x",
            desc=f"base setup with {bottleneck_ratio}x",
            bottleneck_path=f"models/{dataset}_bottleneck_unsupervised_P32_{bottleneck_dim}.pth",
            patch_size=32,
            bottleneck_dim=bottleneck_dim,
            batch_size=384,
            epochs=epochs,
            lr=1e-3,  # not used
            freeze_body=False,
            freeze_head=False,
            freeze_embeddings=False,
            dataset=dataset,
            num_classes=num_classes,
            pre_train=False,
        )

        finetune(experiment_params, device, save_folder)


