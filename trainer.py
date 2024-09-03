from hyperparams import epochs
from callbacks import checkpoint_callback, early_stop_callback
import torch
import lightning as L

trainer = L.Trainer(
    max_epochs=epochs,
    callbacks=[checkpoint_callback, early_stop_callback],
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
)

