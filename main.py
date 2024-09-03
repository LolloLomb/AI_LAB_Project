# IMPORT

from hyperparams import in_channels, num_classes as out_channels, batch_size, train_size_proportion, num_workers, dataset_dir
from customDataLoader import ImageDataModule
from cnn import SimpleCNN
from trainer import trainer as trainer
from lightning.pytorch.tuner import Tuner
# usiamo AdamW, standard dentro la mia Unet2D
import os

LOADFROMCKPT = 0

model = SimpleCNN(in_channels, out_channels)

data_module = ImageDataModule(dataset_dir=dataset_dir, batch_size=batch_size, val_split=1-train_size_proportion, num_workers=num_workers)
data_module.setup()

# Caricamento del checkpoint (se esiste)
def checkpoint_loader():
    checkpoint_path = 'checkpoint'
    if os.path.exists(checkpoint_path):
        try:
            name = os.listdir('checkpoint')[0]
        except IndexError:
            return
        checkpoint_path = os.path.join(checkpoint_path, name)
        model = SimpleCNN.load_from_checkpoint(checkpoint_path)
        print(f"Checkpoint caricato da {checkpoint_path}")
    else:
        print("Nessun checkpoint trovato. Iniziamo un nuovo addestramento.")

if LOADFROMCKPT == 1:
    checkpoint_loader()

tuner = Tuner(trainer)
tuner.lr_find(model, datamodule=data_module)
model.val_acc_list = []

trainer.fit(model, datamodule=data_module)

print(model.learning_rate)
print(model.train_acc_list)
print(model.val_acc_list)
