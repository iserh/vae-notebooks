import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from cifar10_model import VAE
from torchvision.datasets import CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader


vae = VAE(
    input_size=32 * 32,
    z_dim=50,
    beta=1.0 * 50 / (32 * 32),
)

cifar10 = CIFAR10(
    root="~/torch_datasets",
    transform=transforms.ToTensor(),
    train=True,
)

train_loader = DataLoader(cifar10, batch_size=128, shuffle=True, num_workers=24, pin_memory=True)

checkpoint_callback = ModelCheckpoint(
    dirpath="cifar10_checkpoints",
    period=50,
    save_top_k=-1,
)
checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

# Add your callback to the callbacks list
trainer = pl.Trainer(
    callbacks=[checkpoint_callback],
    gpus=1,
    progress_bar_refresh_rate=5,
    max_epochs=500,
)

trainer.fit(
    model=vae,
    train_dataloader=train_loader,
)
