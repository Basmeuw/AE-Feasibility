class Experiment:

    def __init__(self, title="untitled", desc="", patch_size=16, num_classes=100, bottleneck_path=None,
                 embed_dim=768, batch_size=32,epochs = 10, lr = 1e-3, bottleneck_dim=768, freeze_head = False, freeze_body = False, freeze_embeddings = False, save_model=False,
                 dataset = "CIFAR100", bottleneck_type="identity", pre_train = False, pre_train_epochs = 1, pre_train_lr = 1e-3, bottleneck_finetune_lr = 1e-3, body_finetune_lr = 1e-4, min_anneal=5e-5, weight_decay=0.01,
                 val_fraction = 0.1, pre_train_fraction = 0.0):

        self.title = title
        self.desc = desc
        self.patch_size = patch_size
        self.bottleneck_dim = bottleneck_dim
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.bottleneck_path = bottleneck_path

        self.lr = lr

        # self.best_train_loss = float('inf')  # Initialize best training loss to infinity
        # self.best_val_loss = float('inf')
        # self.best_mpe = float('inf')  # Initialize best MPE to infinity
        # self.model_param_count = 0  # Placeholder for model parameter count


        self.batch_size = batch_size  # Default batch size
        self.epochs = epochs


        self.freeze_head = freeze_head
        self.freeze_body = freeze_body
        self.freeze_embeddings = freeze_embeddings
        self.pre_train = pre_train
        self.pre_train_epochs = pre_train_epochs
        self.save_model = save_model
        self.dataset = dataset
        self.bottleneck_type = bottleneck_type

        self.pre_train_lr = pre_train_lr
        self.bottleneck_finetune_lr = bottleneck_finetune_lr
        self.body_finetune_lr = body_finetune_lr
        self.min_anneal = min_anneal
        self.weight_decay = weight_decay

        self.val_fraction = val_fraction
        self.pre_train_fraction = pre_train_fraction



