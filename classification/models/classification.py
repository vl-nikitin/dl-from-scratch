import gc
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

class ClassificationModel(lightning.LightningModule): 
    def __init__(self, criterion, learning_rate=1e-3, scheduler_frequency=2, scheduler_gamma=0.9) -> None:
        super(ClassificationModel, self).__init__()
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.scheduler_frequency = scheduler_frequency
        self.scheduler_gamma = scheduler_gamma
        self.avg_train_loss = 0.0
        self.avg_val_loss = 0.0
        self.avg_test_loss = 0.0
        self.avg_train_accuracy = 0.0
        self.avg_val_accuracy = 0.0
        self.avg_test_accuracy = 0.0
        
    def training_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        Y_pred = self.forward(X_batch)
        loss = self.criterion(Y_pred, Y_batch)
        self.avg_train_loss += loss
        with torch.no_grad():
            res = (Y_pred.argmax(-1) == Y_batch)
        self.avg_train_accuracy += res.sum() / len(Y_batch)

        return loss

    def on_train_epoch_end(self):
        self.avg_train_loss = self.avg_train_loss / self.trainer.num_training_batches
        self.avg_train_accuracy = self.avg_train_accuracy / self.trainer.num_training_batches
        self.logger.experiment.add_scalars('loss', {'train': self.avg_train_loss}, self.current_epoch)
        self.logger.experiment.add_scalars('accuracy', {'train': self.avg_train_accuracy}, self.current_epoch)
        # log only for checkpointing, thus logger=False
        self.log('loss_train', self.avg_train_loss, logger=False)
        self.avg_train_loss = 0.0
        self.avg_train_accuracy = 0.0
        
    def validation_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        Y_pred = self.forward(X_batch)
        loss = self.criterion(Y_pred, Y_batch)
        self.avg_val_loss += loss
        res = (Y_pred.argmax(-1) == Y_batch)
        self.avg_val_accuracy += res.sum() / len(Y_batch)


    def on_validation_epoch_end(self):
        self.avg_val_loss = self.avg_val_loss / self.trainer.num_val_batches[0]
        self.avg_val_accuracy = self.avg_val_accuracy / self.trainer.num_val_batches[0]
        self.logger.experiment.add_scalars('loss', {'validation': self.avg_val_loss}, self.current_epoch)
        self.logger.experiment.add_scalars('accuracy', {'validation': self.avg_val_accuracy}, self.current_epoch)

        # log only for checkpointing, thus logger=False
        self.log('loss_validation', self.avg_val_loss, logger=False)
        
        self.avg_val_loss = 0.0
        self.avg_val_accuracy = 0.0

    def test_step(self, batch, batch_idx):
        X_batch, Y_batch = batch
        Y_pred = self.forward(X_batch)
        loss = self.criterion(Y_pred, Y_batch)
        self.avg_test_loss += loss
        self.avg_test_accuracy += (Y_pred.argmax(-1) == Y_batch).sum() / len(Y_batch)

    def on_test_epoch_end(self):
        test_dataloaders = self.trainer.test_dataloaders
        assert isinstance(test_dataloader, torch.utils.data.DataLoader)
        loss = self.avg_test_loss / len(test_dataloader)
        accuracy = self.avg_val_accuracy / len(test_dataloader)
        self.log('loss_test', loss)
        self.log('accuracy_test', accuracy)
        self.avg_test_loss = 0.0
        self.avg_test_accuracy = 0.0
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=5 * 1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.scheduler_gamma)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "frequency": self.scheduler_frequency
        },
    }

def clean_memory():
    gc.collect()
    lightning.pytorch.utilities.memory.garbage_collection_cuda()
    torch.cuda.empty_cache()

def train(model, train_dataloader, valid_dataloader, max_epochs=20, name='logs', version=None):
    clean_memory()
    # ckpt_callbacks = {
    #     'train_loss': ModelCheckpoint(monitor='loss_train', mode='min'),
    #     'val_loss': ModelCheckpoint(monitor='loss_validation', mode='min')
    # }
    ckpt_callbacks = None

    logger = lightning.pytorch.loggers.tensorboard.TensorBoardLogger('tb_logs', name=name, version=version)
    callbacks_copy = list(ckpt_callbacks.values()) if ckpt_callbacks is not None else None
    trainer = lightning.Trainer(accelerator='gpu', devices=-1, max_epochs=max_epochs, callbacks=callbacks_copy, logger=logger)

    trainer.fit(model, train_dataloader, valid_dataloader)

    return trainer, ckpt_callbacks

def get_datasets(dataset_class, preprocessing=[], train_augmentations=[]):
    dataclass_varnames = dataset_class.__init__.__code__.co_varnames
    assert 'download' in dataclass_varnames
    if preprocessing or train_augmentations:
        assert 'transform' in dataclass_varnames
    train_data_tfs = transforms.Compose([*preprocessing, *train_augmentations])
    val_data_tfs = transforms.Compose(preprocessing)

    root = "./"
    if 'train' in dataclass_varnames:
        train_dataset = dataset_class(root, train=True,  transform=train_data_tfs, download=True)
        val_dataset = dataset_class(root, train=False, transform=val_data_tfs, download=True)
        val_len = len(val_dataset) // 2
        test_len = len(val_dataset) - val_len
        val_dataset, test_dataset  = random_split(val_dataset, [val_len, test_len])
        return train_dataset, val_dataset, test_dataset
    elif 'split' in dataclass_varnames:
        train_dataset = dataset_class(root, split='train', transform=train_data_tfs, download=True)
        val_dataset = dataset_class(root, split='val', transform=val_data_tfs, download=True)
        test_dataset = dataset_class(root, split='test', transform=val_data_tfs, download=True)

        return train_dataset, val_dataset, test_dataset
    else:
        raise TypeError("Dataset class must accept 'train' or 'split' parameter")
        
def get_dataloaders(dataset_class, batch_size=8, size=(224, 224), normalization=[[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]], train_augmentations=[], num_workers=5):
    assert issubclass(dataset_class, Dataset)

    preprocessing = [
        transforms.ToTensor(),
        # tfs.Normalize([0.491, 0.482 ,0.446], [0.247, 0.243, 0.261]),
        transforms.Normalize(*normalization),
        transforms.Resize(size),
    ]

    train_dataset, val_dataset, test_dataset = get_datasets(dataset_class, preprocessing=preprocessing, train_augmentations=train_augmentations)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, valid_dataloader, test_dataloader