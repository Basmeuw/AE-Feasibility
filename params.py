class Experiment:

    def __init__(self, title="untitled", patch_size=16, num_classes=100, bottleneck_path=None,
                 embed_dim=768, batch_size=32,epochs = 10, lr = 1e-3, bottleneck_dim=384, freeze_head = False, freeze_body = True):

        self.title = title
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




