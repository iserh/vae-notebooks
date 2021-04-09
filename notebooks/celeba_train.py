import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from celeba_model import VAE
from torchvision.datasets import CelebA
from torchvision import transforms
from torch.utils.data import DataLoader


vae = VAE(
    input_size=64 * 64,
    z_dim=50,
    beta=1.0 * 50 / (64 * 64),
)

celeba = CelebA(
    root="~/torch_datasets",
    transform=transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
        ]
    ),
    split="train",
    download=False,
    target_type="identity",
)

train_loader = DataLoader(celeba, batch_size=128, shuffle=True, num_workers=24, pin_memory=True)

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    period=50,
)

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
