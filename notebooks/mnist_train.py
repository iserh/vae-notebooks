import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from mnist_model import VAE
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader


vae = VAE(
    input_size=28 * 28,
    z_dim=2,
    beta=1.0,
)

mnist = MNIST(
    root="~/torch_datasets",
    transform=transforms.ToTensor(),
    train=True,
)

train_loader = DataLoader(mnist, batch_size=64, shuffle=True, num_workers=24, pin_memory=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=f"mnist_checkpoints/z_dim={vae.z_dim}",
    period=25,
    save_top_k=-1,
)
checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"

# Add your callback to the callbacks list
trainer = pl.Trainer(
    callbacks=[checkpoint_callback],
    gpus=1,
    progress_bar_refresh_rate=5,
    max_epochs=100,
)

trainer.fit(
    model=vae,
    train_dataloader=train_loader,
)
