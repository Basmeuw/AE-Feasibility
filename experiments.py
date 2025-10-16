import os

import torch

from bottleneck import Bottleneck, SingleLinearBottleneck
from main import finetune
from params import Experiment


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
    elif dataset == "StanfordCars":
        num_classes = 196
    elif dataset == "SVHN":
        num_classes = 10
    else:
        print(f"Dataset {dataset} not supported.")
        return -1
    return num_classes

# Used for the scenario where we pre-train the bottleneck on a single dataset only
def pre_train_bottleneck(dataset, bottleneck_type, bottleneck_dims, data_fractions, device):
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

    from unsupervised import  train_bottleneck_unsupervised
    for bottleneck_dim in bottleneck_dims:
        for data_fraction in data_fractions:
            print(f"Pre-training, bottleneck dim {bottleneck_dim}, data_fraction {data_fraction}, for dataset {dataset}")
            train_bottleneck_unsupervised(dataset, bottleneck_type, data_fraction, f"{dataset}_{data_fraction}_bottleneck_unsupervised_P32", f"activations_{dataset}", device,
                                          bottleneck_dim=bottleneck_dim, epochs=50)

# Used for the scenario where we pre-train the bottleneck on a single dataset only. Here we finetune with the generated datasets
def transfer_bottleneck_data_fraction(pre_train_dataset, dataset, epochs, bottleneck_dims, data_fractions, device, save_folder):
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



def transfer_bottleneck_data_fraction_general(experiment_folder_name, is_retrieve_activations, is_pre_train_bottleneck, is_train, include_baseline, pre_train_dataset, train_dataset, epochs, bottleneck_dims, data_fractions, device):

    bottleneck_type = "bottleneck"

    if is_retrieve_activations:
        retrieval_params = Experiment(
            title=f"{pre_train_dataset}",
            desc="retrieve activations",
            bottleneck_path=None,
            patch_size=32,
            batch_size=64,
            num_classes=get_num_classes(dataset=pre_train_dataset),
            dataset=pre_train_dataset,
        )
        from unsupervised import  retrieve_activations
        retrieve_activations(retrieval_params, device)

    if is_pre_train_bottleneck:
        # First pre_train the bottleneck with the possible data fractions
        pre_train_bottleneck(pre_train_dataset, bottleneck_type, bottleneck_dims, data_fractions, device)

    # # Create a randomly initialized bottleneck
    # if bottleneck_dims contains 0
    if 0 in bottleneck_dims:
        for bottleneck_dim in bottleneck_dims:
            model = SingleLinearBottleneck(768, bottleneck_dim)
            torch.save({
                'model_state_dict': model.state_dict(),
            }, f'models/{pre_train_dataset}_0_bottleneck_unsupervised_P32_{bottleneck_dim}.pth') # HARDCODED


    if is_train:
        if include_baseline:
            experiment_compression_vs_accuracy_general(train_dataset, device, epochs=epochs, baseline_only=True,
                                                       save_folder=experiment_folder_name)
        transfer_bottleneck_data_fraction(pre_train_dataset, train_dataset, epochs, bottleneck_dims, data_fractions, device, save_folder=experiment_folder_name)


    from plots import plot_metric_from_runs, plot_final_metric_bar_chart
    plot_metric_from_runs(
        parent_folder=experiment_folder_name,
        metric_name="val_losses",
        title=f"{pre_train_dataset} to {train_dataset} Transfer Val Losses",
        save_path="val_loss"
    )

    plot_metric_from_runs(
        parent_folder=experiment_folder_name,
        metric_name="val_accuracies",
        title=f"{pre_train_dataset} to {train_dataset} Transfer Val Accuracy",
        save_path="val_acc"
    )

    # 2. Plot the final test accuracy (Bar Chart)
    plot_final_metric_bar_chart(
        parent_folder=experiment_folder_name,
        final_metric_key="final_test_accuracy",
        title=f"{pre_train_dataset} to {train_dataset} Transfer Final Test acc",
        save_path="final_test"
    )



def experiment_compression_vs_accuracy_general(pretrain_dataset, dataset, data_fraction, bottleneck_dims, epochs, bottleneck_types: list, device, baseline_only=False, save_folder="runs/test"):
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

    #
    # retrieve_activations(retrieval_params, device)
    #
    # train_bottleneck_unsupervised(f"{dataset}_bottleneck_unsupervised_P32", f"activations_{dataset}", device, bottleneck_dim=48, epochs=50)
    # train_bottleneck_unsupervised(f"{dataset}_bottleneck_unsupervised_P32", f"activations_{dataset}", device, bottleneck_dim=96, epochs=50)
    # train_bottleneck_unsupervised(f"{dataset}_bottleneck_unsupervised_P32", f"activations_{dataset}", device, bottleneck_dim=192, epochs=50)
    # train_bottleneck_unsupervised(f"{dataset}_bottleneck_unsupervised_P32", f"activations_{dataset}", device, bottleneck_dim=384, epochs=50)

    for bottleneck_type in bottleneck_types:
        for bottleneck_dim in bottleneck_dims:
            # check if file exists
            file_path = f"models/{pretrain_dataset}_{bottleneck_type}_unsupervised_P32_{bottleneck_dim}.pth"
            if not os.path.isfile(file_path):
                print(f"File {file_path} does not exist. Pre-training now.")
                from unsupervised import train_bottleneck_unsupervised
                print(f"Pre-training, bottleneck dim {bottleneck_dim}, for dataset {pretrain_dataset}")
                train_bottleneck_unsupervised(pretrain_dataset, bottleneck_type, data_fraction,
                                              f"{pretrain_dataset}_{bottleneck_type}_unsupervised_P32", f"activations_{pretrain_dataset}",
                                              device,
                                              bottleneck_dim=bottleneck_dim, epochs=50)
            else:
                print(f"File {file_path} exists. Skipping pre-training.")


    # BASELINE
    experiment_params = Experiment(
        title=f"{dataset}_baseline",
        desc=f"baseline on {dataset}",
        bottleneck_path=None,
        bottleneck_type="identity",
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

    # finetune(experiment_params, device, save_folder)

    if baseline_only:
        return

    for bottleneck_type in bottleneck_types:
        for bottleneck_dim in bottleneck_dims:
            bottleneck_ratio = 768 / bottleneck_dim
            experiment_params = Experiment(
                title=f"{dataset}_bottleneck_{bottleneck_ratio}x",
                desc=f"base setup with {bottleneck_ratio}x",
                bottleneck_path=f"models/{dataset}_{bottleneck_type}_unsupervised_P32_{bottleneck_dim}.pth",
                patch_size=32,
                bottleneck_type=bottleneck_type,
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

    from plots import plot_metric_from_runs, plot_final_metric_bar_chart
    plot_metric_from_runs(
        parent_folder=save_folder,
        metric_name="val_losses",
        title=f"{dataset} with {bottleneck_types} Val Losses",
        save_path="val_loss"
    )

    plot_metric_from_runs(
        parent_folder=save_folder,
        metric_name="val_accuracies",
        title=f"{dataset} with {bottleneck_types} Val Accuracy",
        save_path="val_acc"
    )

    # 2. Plot the final test accuracy (Bar Chart)
    plot_final_metric_bar_chart(
        parent_folder=save_folder,
        final_metric_key="final_test_accuracy",
        title=f"{dataset} with {bottleneck_types} Transfer Final Test acc",
        save_path="final_test"
    )

