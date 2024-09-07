# IMPORT

from hyperparams import in_channels, num_classes as out_channels, batch_size, train_size_proportion, num_workers, dataset_dir, LOADFROMCKPT, TESTPHASE
from customDataLoader import ImageDataModule
from cnn import SimpleCNN
from preprocess import preprocess_single
from trainer import trainer as trainer
from PIL import Image
import torch
from torchvision import transforms # usato per l'immagine di test
from lightning.pytorch.tuner import Tuner
import os
import sys

# Inizializza il dataloader
data_module = ImageDataModule(dataset_dir=dataset_dir, batch_size=batch_size, val_split=1-train_size_proportion, num_workers=num_workers)
data_module.setup()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Caricamento del modello
def checkpoint_loader():
    checkpoint_path = 'checkpoint'
    if os.path.exists(checkpoint_path):
        try:
            name = os.listdir('checkpoint')[0]
        except IndexError:
            return
        checkpoint_path = os.path.join(checkpoint_path, name)
        
        # Carica il modello dal checkpoint
        model = SimpleCNN.load_from_checkpoint(checkpoint_path, 
                                               in_channels=in_channels, 
                                               num_classes=out_channels, 
                                               classes=data_module.classes)  # Passa le classi
        model.to(device)
        print(f"\nCheckpoint caricato da {checkpoint_path}")
        return model
    else:
        print("\nNessun checkpoint trovato. Iniziamo un nuovo addestramento.")
        return None


def main(path = None):
    # Se `LOADFROMCKPT` è impostato, carica il checkpoint
    if LOADFROMCKPT == 1:
        model = checkpoint_loader()
    else:
        model = SimpleCNN(in_channels=in_channels, classes=data_module.classes, num_classes=out_channels)

    # Se siamo nella fase di test
    if TESTPHASE == 1:
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        processed_image_output_path = preprocess_single('test_samples', os.path.basename(path))
        proc = Image.open(processed_image_output_path)
        proc_tensor = test_transform(proc)
        proc_tensor = torch.unsqueeze(proc_tensor, 0).to(device)

        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            probabilities, class_probabilities = model(proc_tensor)  # Ora il modello restituisce probabilità e dict
            # Stampa le probabilità per ogni classe
            for class_name, prob in class_probabilities.items():
                print(f"{class_name} : {100 * float(prob.item()):.2f}%")
    else:
        # Se `LOADFROMCKPT` è 0, cerchiamo il learning rate ottimale e facciamo il training
        if LOADFROMCKPT == 0:
            tuner = Tuner(trainer)
            tuner.lr_find(model, datamodule=data_module)
            model.val_acc_list = []
        
        # Addestra il modello
        trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        main(file_path)
    else:
        print("Nessun file specificato.")

