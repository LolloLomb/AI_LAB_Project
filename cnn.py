import torch
import torch.nn as nn
import lightning as L
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from hyperparams import num_classes as output_channels, filters, learning_rate

class SimpleCNN(L.LightningModule):
    def __init__(self, in_channels, num_classes = output_channels, loss_fx = nn.CrossEntropyLoss(), optimizer=None, learning_rate = learning_rate):
        super().__init__()
        self.loss_fx = loss_fx
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.in_channels = in_channels
        self.out_channels = num_classes
        self.train_acc_list = []
        self.val_acc_list = []
        self.val_labels = []
        self.val_preds = []

        # Definisci una semplice CNN
        self.conv1 = nn.Conv2d(self.in_channels, filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters*2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(filters*2, filters*4, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(filters*4 * 8 * 8, filters*8)  # Assuming input images are 64x64
        self.fc2 = nn.Linear(filters*8, self.out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Definisci il forward pass
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss_fx(y_out, y)
        acc = (y_out.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_out = self.forward(x)
        loss = self.loss_fx(y_out, y)
        acc = (y_out.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)

        # Accumula le etichette e le previsioni per la matrice di confusione
        self.val_labels.extend(y.cpu().numpy())
        self.val_preds.extend(y_out.argmax(dim=1).cpu().numpy())

        return {'val_loss': loss, 'val_acc': acc}
        
    def on_validation_epoch_end(self):
        avg_train_acc = self.trainer.callback_metrics.get('train_acc')
        avg_val_acc = self.trainer.callback_metrics.get('val_acc')
        if avg_val_acc is not None:
            self.val_acc_list.append(avg_val_acc.item())
        if avg_train_acc is not None:
            self.train_acc_list.append(avg_train_acc.item())

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)     

    def configure_optimizers(self):
        if self.optimizer is None:
            return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        else:
            return self.optimizer
    
    def on_train_end(self):
    # Plot dell'accuratezza in funzione delle epoche
        plt.figure(figsize=(10, 5))  # Crea una nuova figura con dimensioni personalizzate
        plt.plot(self.val_acc_list, label='Validation Accuracy')
        plt.plot(self.train_acc_list, label='Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epoch')
        plt.legend()
        plt.savefig('acc_over_epoch.jpg')    
        plt.close()  # Chiude la figura corrente per evitare conflitti

        cm = confusion_matrix(self.val_labels, self.val_preds, labels=list(range(self.num_classes)))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=range(self.num_classes))

        plt.figure()  # Crea una nuova figura
        disp.plot()
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.jpg')
        plt.close()  # Chiude la figura corrente per evitare conflitti
