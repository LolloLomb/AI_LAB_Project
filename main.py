# IMPORT

from hyperparams import in_channels, num_classes as out_channels, batch_size, train_size_proportion, num_workers, dataset_dir
from customDataLoader import ImageDataModule
from cnn import SimpleCNN
from preprocess import preprocess_single
from trainer import trainer as trainer
from PIL import Image
import torch
from torchvision import transforms # usato per l'immagine di test
from lightning.pytorch.tuner import Tuner
# usiamo AdamW, standard dentro la mia Unet2D
import os

LOADFROMCKPT = 1
TESTPHASE = 1

model = SimpleCNN(in_channels, out_channels)

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
        model.in_channels = in_channels
        model.num_classes = out_channels
        print(f"Checkpoint caricato da {checkpoint_path}")
    else:
        print("Nessun checkpoint trovato. Iniziamo un nuovo addestramento.")

if LOADFROMCKPT == 1:
    checkpoint_loader()
        
if TESTPHASE == 0:
    data_module = ImageDataModule(dataset_dir=dataset_dir, batch_size=batch_size, val_split=1-train_size_proportion, num_workers=num_workers)
    data_module.setup()

else:
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_path = os.path.join('test_samples', 'test_img.webp')
    
    proccesed_image_output_path = preprocess_single('test_samples', image_path.split("/")[-1], "webp")
    proc = Image.open(proccesed_image_output_path)
    proc_tensor = test_transform(proc)
    proc_tensor = torch.unsqueeze(proc_tensor, 0)
    print(proc_tensor.shape)

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for inference
        output = model(proc_tensor)
        # Process the output as needed
    print(output)

if LOADFROMCKPT == 0:
    tuner = Tuner(trainer)
    tuner.lr_find(model, datamodule=data_module)
    model.val_acc_list = []

    trainer.fit(model, datamodule=data_module)

