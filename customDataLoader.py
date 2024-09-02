# Carica il dataset
import torch.utils
import torch.utils.data
from torchvision import datasets
from torch.utils.data import random_split, DataLoader
from transforms import train_transforms, test_transforms
import lightning as L

L.seed_everything(42)

class ImageDataModule(L.LightningDataModule):
    def __init__(self, dataset_dir, batch_size, val_split, num_workers):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_transforms = train_transforms
        self.val_transforms = test_transforms

        # Carica il dataset completo
        full_dataset = datasets.ImageFolder(self.dataset_dir)

        # Calcola le dimensioni dei sottoinsiemi di train e val
        total_size = len(full_dataset)
        val_size = int(self.val_split * total_size)
        train_size = total_size - val_size

        # Suddividi il dataset
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])
        self.train_dataset = TrDataset(self.train_dataset, train_transforms)
        self.val_dataset = TrDataset(self.val_dataset, test_transforms)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

class TrDataset(torch.utils.data.Dataset):
    # Ho bisogno di applicare le trasformazioni separatamente per train e val
    def __init__(self, base_dataset, transformations):
        super(TrDataset, self).__init__()
        self.base = base_dataset
        self.transformations = transformations

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return self.transformations(x), y
        
'''
def show_images(images, title):
    plt.figure(figsize=(12, 8))
    grid_img = np.transpose(torchvision.utils.make_grid(images, nrow=8, padding=2, normalize=True), (1, 2, 0))
    plt.imshow(grid_img)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Carica un batch di immagini
train_loader = data_module.train_dataloader()
images, labels = next(iter(train_loader))

# Visualizza le immagini prima delle trasformazioni (originali)
show_images(images, title="Immagini dopo le trasformazioni (ColorJitter, Flip(s), Normalize)")
'''
