# IMPORT

from hyperparams import in_channels, num_classes as out_channels, batch_size, train_size_proportion, num_workers, dataset_dir
from customDataLoader import ImageDataModule
from cnn import SimpleCNN
from trainer import trainer as trainer
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

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import lightning as L
import torch

# Configurazione dei dati
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),  # Media e deviazione standard per il dataset Fashion MNIST
    transforms.Resize((64,64))
])

# Carica il dataset Fashion MNIST
train_dataset = datasets.FashionMNIST(root='./data', train=True, transform=transform, download=True)
val_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)

# Crea i dataloader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=7)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=7)

# Definisci il modello, i logger e il trainer
model = SimpleCNN(in_channels=1, num_classes=10)  # Fashion MNIST ha 10 classi e immagini in scala di grigi

# Configura il trainer di PyTorch Lightning
trainer = L.Trainer(
    max_epochs=1,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
)

# Allena il modello
trainer.fit(model, train_loader, val_loader)

# Dopo l'allenamento, visualizza i risultati
model.on_train_end()


#trainer.fit(model, datamodule=data_module)
