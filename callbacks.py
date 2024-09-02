from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

# Configurazione dei callbacks
checkpoint_callback = ModelCheckpoint(
    monitor='val_acc',
    mode='max',
    save_top_k=1,
    dirpath='checkpoint',
    filename='best-checkpoint-{epoch:02d}-{val_acc:.2f}',
    verbose=True
)

early_stop_callback = EarlyStopping(
    monitor='val_acc',
    patience=10,
    mode='max',
    verbose=True
)