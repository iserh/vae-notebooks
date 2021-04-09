import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from tqdm import trange, tqdm
from pathlib import Path
import pandas as pd
import random
from torchvision.datasets import MNIST
import torchvision.utils as vutils
import torchvision.transforms as transforms
from mnist_model import VAE
from matplotlib.pyplot import plot, scatter, savefig, close, figure, title, subplot, imshow, axis, grid, legend
import numpy as np

# set seed
model_seed = 936
data_seed =  83
# set cudnn backend to be deterministic
torch.backends.cudnn.deterministic = True
# utility function for seeding
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
# for plotting
def ordering(rows, cols):
    # this permutation is used for ordering the images - for visualization purpose
    return torch.cat([torch.LongTensor([i + rows * j for j in range(cols)]) for i in range(rows)], dim=0)

for data_seed in [1033, 1337]:
    for i in [2, 3, 4, 5, 10, 20, 30, 50, 100, 200, 500, 1000, 2000]:

        # data hyperparameters
        n_originals_per_class =     i
        n_generated_per_class =     i
        # vae hyperparamters
        vae_z_dim =                 50
        vae_beta =                  1.0
        # gc hyperparameters
        gc_training_steps =         800
        gc_batch_size =             16
        gc_std =                    1.8
        gen_neighbors =             (1 if i == 2 else 2) if i <= 3 else 3
        gen_alpha =                 0.4
        gen_std =                   0.4
        # cnn model hyperparameters
        cls_training_steps =        2000
        cls_batch_size =            32
        cls_eval_interval =         20


        # In[12]:


        # cuda
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print("Using device:", device)
        # output dir for plots
        plot_dir = Path(f"../plots/MNIST/runs/{n_originals_per_class}-{n_generated_per_class}-{vae_z_dim}-{vae_beta}-{gc_training_steps}-{gc_batch_size}-{gc_std}-{gen_neighbors}-{gen_alpha}-{cls_training_steps}-{cls_batch_size}-{cls_eval_interval}")
        plot_root = Path(f"../plots/MNIST")
        plot_root.mkdir(exist_ok=True, parents=True)
        plot_dir.mkdir(exist_ok=True, parents=True)


        # ## Data Preparation
        # 

        # In[16]:


        # load mnist
        mnist_train = MNIST(
            root="~/torch_datasets",
            transform=transforms.ToTensor(),
            train=True,
        )
        mnist_test = MNIST(
            root="~/torch_datasets",
            transform=transforms.ToTensor(),
            train=False,
        )

        # extract test data
        x_test, y_test = zip(*[(x, y) for x, y in torch.utils.data.DataLoader(mnist_test, batch_size=512, num_workers=4)])
        x_test, y_test = torch.cat(x_test, dim=0), torch.cat(y_test, dim=0)

        # extract train data
        x_train, y_train = zip(*[(x, y) for x, y in torch.utils.data.DataLoader(mnist_train, batch_size=512, num_workers=4)])
        x_train, y_train = torch.cat(x_train, dim=0), torch.cat(y_train, dim=0)

        # remember the shape of the data
        input_shape = x_train.size()[1:]
        n_classes = len(torch.unique(y_train))

        # separate train data into the different classes
        train_per_label = [x_train[y_train == i] for i in range(n_classes)]

        # following data operations include random permutations so seed everything
        seed_everything(data_seed)
        # reduce training size
        train_per_label = [X[np.random.permutation(X.size(0))[:n_originals_per_class]] for X in train_per_label]


        # ## Variational Auto-Encoder

        # In[ ]:


        vae = VAE.load_from_checkpoint("mnist_checkpoints/beta=1.0/epoch=99-step=93799.ckpt",
            input_size=np.prod(input_shape), z_dim=vae_z_dim, beta=vae_beta
        ).eval()
        early_vae = VAE.load_from_checkpoint("mnist_checkpoints/beta=1.0/epoch=24-step=23449.ckpt",
            input_size=np.prod(input_shape), z_dim=vae_z_dim, beta=vae_beta
        ).eval()


        # In[ ]:


        def generate_examples(n):
            seed_everything(data_seed)
            global vae
            # work on device
            vae.to(device)
            with torch.no_grad():
                z_per_label, log_v_per_label = [*zip(*[vae.encoder(x.to(device)) for x in train_per_label])]
                z_per_label, log_v_per_label = torch.stack(z_per_label, dim=0).cpu(), torch.stack(log_v_per_label, dim=0).cpu()
            
            # generate some samples
            rand_idx = torch.randint(0, n_originals_per_class, size=(n_classes, n))
            z_sample_per_label = torch.stack([z[idx] for z, idx in (zip(z_per_label, rand_idx))], dim=0)
            log_v_rand_per_label = torch.stack([log_v[idx] for log_v, idx in (zip(log_v_per_label, rand_idx))], dim=0)
            x_sample_per_label = torch.stack([z[idx] for z, idx in (zip(train_per_label, rand_idx))], dim=0)
            
            new_z_rand_per_label, partners_per_label = [], []
            x_origs_per_label = []
            z_origs_per_label, z_partners_per_label = [], []
            for z_orig, z_sample, x, x_sample in zip(z_per_label, z_sample_per_label, train_per_label, x_sample_per_label):
                # build nearest neighbour tree, k + 1 because the first neighbour is the point itself
                nbrs = NearestNeighbors(n_neighbors=gen_neighbors + 1, algorithm="ball_tree").fit(z_orig)
                # get indices of k nearest neighbours for each latent vector
                _, indices = nbrs.kneighbors(z_sample)
                # the new latents and their root latents
                new_latents, partners = [], []
                z_origs, z_partners = [], []
                x_origs = []
                for i, idx in enumerate(indices):
                    # choose one of the k nearest neighbors (ignore first one because its the latent vector itself)
                    neighbor_idx = np.random.choice(idx[1:])
                    # interpolate
                    new_latents.append((z_orig[neighbor_idx] - z_sample[i]) * gen_alpha + z_sample[i] + torch.empty_like(z_orig[neighbor_idx]).normal_(0, gen_std))
                    # new_latents.append((z_orig[neighbor_idx] - z_sample[i]) * gen_alpha + z_sample[i])
                    # root is the neighbor
                    partners.append(x[neighbor_idx])  
                    z_origs.append(z_sample[i])
                    z_partners.append(z_orig[neighbor_idx])
                    x_origs.append(x_sample[i])
            
                new_z_rand_per_label.append(torch.stack(new_latents, dim=0))
                partners_per_label.append(torch.stack(partners, dim=0))
                x_origs_per_label.append(torch.stack(x_origs, dim=0))
                
                z_origs_per_label.append(torch.stack(z_origs))
                z_partners_per_label.append(torch.stack(z_partners))
            
            new_z_rand_per_label = torch.stack(new_z_rand_per_label, dim=0)

            # build a dataset with both the original samples
            # and also some generated ones using the vaes
            z_rand_ds = torch.utils.data.TensorDataset(new_z_rand_per_label)
            generated_per_label = []
            with torch.no_grad():
                for (z_rand_per_label,) in torch.utils.data.DataLoader(z_rand_ds, batch_size=512):
                    generated_per_label.append(torch.stack([
                        vae.decoder.forward(z_rand.to(device))
                        for z_rand in z_rand_per_label
                    ], dim=0))
            # move vae back to cpu
            vae.to('cpu')
            # return [torch.cat([*gen], dim=0).cpu() for gen in zip(*generated_per_label)], partners_per_label, z_origs_per_label, z_partners_per_label, new_z_rand_per_label, x_origs_per_label
            return [torch.cat([*gen], dim=0).cpu() for gen in zip(*generated_per_label)]

        # x_gen_per_label, partners_per_label, _, _, _, _ = generate_examples(10)
        # x_gen_per_label = torch.cat(x_gen_per_label, dim=0)
        # partners_per_label = torch.cat(partners_per_label, dim=0)
        # 
        # # plot original and fake images
        # figure(tight_layout=True, figsize=(10, 5))
        # subplot(121)
        # axis("off")
        # title("Partner")
        # imshow(
        #     np.transpose(
        #         vutils.make_grid(partners_per_label[ordering(10, n_classes)], padding=5, normalize=True, nrow=10),
        #         (1, 2, 0),
        #     ),
        # )
        # subplot(122)
        # axis("off")
        # title("Fake")
        # imshow(
        #     np.transpose(
        #         vutils.make_grid(x_gen_per_label[ordering(10, n_classes)], padding=5, normalize=True, nrow=10),
        #         (1, 2, 0),
        #     ),
        # )
        # savefig(plot_dir / "original-fake.pdf")
        # close()


        # In[ ]:


        # x_gen_per_label, partners_per_label, z_orig_per_label, z_partners_per_label, new_z_rand_per_label, x_origs_per_label = generate_examples(1)
        # 
        # from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        # def imscatter(x, y, image, ax=None, zoom=1):
        #     if ax is None:
        #         ax = plt.gca()
        #     im = OffsetImage(image, zoom=zoom)
        #     x, y = np.atleast_1d(x, y)
        #     artists = []
        #     for x0, y0 in zip(x, y):
        #         ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        #         artists.append(ax.add_artist(ab))
        #     ax.update_datalim(np.column_stack([x, y]))
        #     ax.autoscale()
        #     return artists
        # 
        # 
        # fig = figure(tight_layout=True, figsize=(10, 10))
        # cmap = get_cmap('tab10')
        # 
        # for i, (z_orig, z_partner, z_new) in enumerate(zip(z_orig_per_label, z_partners_per_label, new_z_rand_per_label)):
        #     scatter(*z_orig.T, s=10, label=i, color=cmap(i))
        #     scatter(*z_partner.T, s=10, color=cmap(i))
        #     scatter(*z_new.T, s=10, color=cmap(i))
        # 
        # idx = 7
        # 
        # z_orig = z_orig_per_label[idx].squeeze(0)
        # z_new = new_z_rand_per_label[idx].squeeze(0)
        # z_partner = z_partners_per_label[idx].squeeze(0)
        # 
        # imscatter(*z_orig    * 1.00 + torch.Tensor([0.15, -0.15]).T, x_origs_per_label[idx].squeeze(0).squeeze(0), zoom=1.4)
        # imscatter(*z_new     * 1.10 + torch.Tensor([0, 0.1]).T, x_gen_per_label[idx].squeeze(0).squeeze(0), zoom=1.4)
        # imscatter(*z_partner * 0.9  + torch.Tensor([0, 0]).T, partners_per_label[idx].squeeze(0).squeeze(0), zoom=1.4)
        # 
        # 
        # idx = 1
        # 
        # z_orig = z_orig_per_label[idx].squeeze(0)
        # z_new = new_z_rand_per_label[idx].squeeze(0)
        # z_partner = z_partners_per_label[idx].squeeze(0)
        # 
        # imscatter(*z_orig    * 1.00 + torch.Tensor([0.15, 0.15]).T, x_origs_per_label[idx].squeeze(0).squeeze(0), zoom=1.4)
        # imscatter(*z_new     * 1.10 + torch.Tensor([0, 0.1]).T, x_gen_per_label[idx].squeeze(0).squeeze(0), zoom=1.4)
        # imscatter(*z_partner * 0.85 + torch.Tensor([0, 0]).T, partners_per_label[idx].squeeze(0).squeeze(0), zoom=1.4)
        # 
        # 
        # 
        # idx = 6
        # 
        # z_orig = z_orig_per_label[idx].squeeze(0)
        # z_new = new_z_rand_per_label[idx].squeeze(0)
        # z_partner = z_partners_per_label[idx].squeeze(0)
        # 
        # imscatter(*z_orig    * 1.00 + torch.Tensor([0.15, -0.15]).T, x_origs_per_label[idx].squeeze(0).squeeze(0), zoom=1.4)
        # imscatter(*z_new     * 1.10 + torch.Tensor([0, 0.1]).T, x_gen_per_label[idx].squeeze(0).squeeze(0), zoom=1.4)
        # imscatter(*z_partner * 0.85 + torch.Tensor([0, 0]).T, partners_per_label[idx].squeeze(0).squeeze(0), zoom=1.4)
        # 
        # # ylim(-0.5, 2.5)
        # # xlim(-2, 0.5)
        # 
        # legend()
        # savefig(plot_dir / "interpolation.pdf")
        # close()


        # #### Visualize the Feature Space

        # In[ ]:


        dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=512)
        with torch.no_grad():
            z, y = [*zip(*[(vae.encoder(x)[0], y) for x, y in tqdm(dl)])]
            z, y = torch.cat(z, dim=0).cpu(), torch.cat(y, dim=0).cpu()

        figure(tight_layout=True, figsize=(5, 5))
        for i in range(n_classes):
            scatter(*PCA(2).fit_transform(z[y == i]).T, s=2, label=i)
        legend()
        savefig(plot_dir / f"feature_space.pdf")
        close()


        # #### Visualize some generated images

        # In[ ]:


        # original images
        orig_img = torch.cat([x[:10] for x in train_per_label], dim=0)

        x_gen = torch.cat(generate_examples(10), dim=0)

        # plot original and fake images
        figure(tight_layout=True, figsize=(10, 5))
        subplot(121)
        axis("off")
        title("Original")
        imshow(
            np.transpose(
                vutils.make_grid(orig_img[ordering(min(n_originals_per_class, 10), n_classes)], padding=5, normalize=True, nrow=10),
                (1, 2, 0),
            ),
        )
        subplot(122)
        axis("off")
        title("Fake")
        imshow(
            np.transpose(
                vutils.make_grid(x_gen[ordering(min(n_originals_per_class, 10), n_classes)], padding=5, normalize=True, nrow=10),
                (1, 2, 0),
            ),
        )
        savefig(plot_dir / "original-fake.pdf")
        close()


        # ## Generative Classifier

        # In[ ]:


        class GenerativeClassifier(nn.Module):
            def __init__(self):
                super(GenerativeClassifier, self).__init__()
                self.conv1 = nn.Conv2d(1, 2, kernel_size=5)
                self.conv2 = nn.Conv2d(2, 4, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(4 * 4 * 4, 8)
                self.fc2 = nn.Linear(8, 1)
            def forward(self, x):
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                x = x.view(-1, 4 * 4 * 4)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, training=self.training)
                x = self.fc2(x)
                return torch.sigmoid(x)


        # ### Generating the data for GC Training

        # In[ ]:


        # seed before generating the noisy examples for training the generative classifier
        seed_everything(data_seed)

        # build dataset of fake and real samples
        real_x = torch.cat(train_per_label, dim=0)
        with torch.no_grad():
            # create random latents outside the normal distribution (0, 1)
            z_rand = torch.empty((n_classes, n_originals_per_class, vae_z_dim)).normal_(0, gc_std)
            fake_x = torch.cat([
                early_vae.decoder.forward(z)
                for z in z_rand
            ], dim=0)
            
        # create labels
        real_y = torch.ones(real_x.size(0))
        fake_y = torch.zeros(fake_x.size(0))
        # pack into dataset
        gc_dataset = torch.utils.data.TensorDataset(
            torch.cat((real_x, fake_x), dim=0),
            torch.cat((real_y, fake_y), dim=0)
        )

        # visualize the latents used for generating noisy examples
        figure(figsize=(20, 10), tight_layout=True)
        subplot(121)
        title("Noisy Latents")
        scatter(*PCA(2).fit_transform(z_rand.view(-1, vae_z_dim)).T)

        subplot(122)
        title("Noisy examples")
        axis("off")
        imshow(
            np.transpose(
                vutils.make_grid(fake_x[:100][ordering(min(n_originals_per_class, 10), n_classes)], padding=5, normalize=True, nrow=10),
                (1, 2, 0),
            ),
        )
        close()


        # ### Training the Generative Classifier

        # In[ ]:


        # seed before training the generative classifier
        seed_everything(model_seed)

        # train generative classifier
        gc = GenerativeClassifier()
        gc.to(device).train()
        optim = torch.optim.Adam(gc.parameters())

        seed_everything(data_seed)
        t = trange(gc_training_steps)
        for i in t:
            # get batch to train on
            batch_idx = np.random.randint(0, len(gc_dataset), gc_batch_size)
            x, y_hat = gc_dataset[batch_idx]
            # apply model and compute loss
            y = gc.forward(x.to(device)).flatten()
            loss = F.binary_cross_entropy(y, y_hat.to(device))  # + 0.05 * (1 - y).mean()
            # update model parameters
            optim.zero_grad()
            loss.backward()
            optim.step()

        # move classifier back to cpu
        # and set it into evaluation mode
        gc = gc.to('cpu').eval()


        # #### Visualize the behaviour of the Generative Classifier

        # In[ ]:


        seed_everything(data_seed)

        x_gen = torch.cat(generate_examples(50), dim=0)
        with torch.no_grad():
            # apply generative classifier
            mask = gc.forward(x_gen) > 0.5
        # convert to numpy
        x_gen = x_gen
        mask = mask
        # apply mask
        x_good, x_bad = x_gen.clone(), x_gen.clone()
        x_good[~mask], x_bad[mask] = 0, 0

        # visualize the choice of
        # the generative classifier
        figure(figsize=(5, 13), tight_layout=True)
        subplot(121)
        axis("off")
        title("Good Samples")
        imshow(
            np.transpose(
                vutils.make_grid(x_good[ordering(50, n_classes)], padding=5, normalize=True, nrow=10),
                (1, 2, 0),
            ),
        )
        subplot(122)
        axis("off")
        title("Bad Samples")
        imshow(
            np.transpose(
                vutils.make_grid(x_bad[ordering(50, n_classes)], padding=5, normalize=True, nrow=10),
                (1, 2, 0),
            ),
        )
        close()


        # ## Classification Task

        # In[7]:


        class Classifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
                self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
                self.conv2_drop = nn.Dropout2d()
                self.fc1 = nn.Linear(4 * 4 * 16, 64)
                self.fc2 = nn.Linear(64, 10)
            def forward(self, x):
                x = F.relu(F.max_pool2d(self.conv1(x), 2))
                x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
                x = x.view(-1, 4 * 4 * 16)
                x = F.relu(self.fc1(x))
                x = F.dropout(x, training=self.training)
                x = self.fc2(x)
                return F.log_softmax(x, dim=-1)


        # ### Training script for the Classification Model

        # In[8]:


        def train_classifier(dataset):
            seed_everything(model_seed)
            # create a cnn model
            model = Classifier()
            # move model to device and set
            # it in train mode
            model.to(device).train()
            optim = torch.optim.Adam(model.parameters())
            # values we want to track
            train_losses, test_losses = [], []
            weighted_f1_scores, acc_scores = [], []
            
            seed_everything(data_seed)
            # train loop
            t = trange(cls_training_steps)
            for i in t:
                # get batch to train from
                x_idx = np.random.randint(0, len(dataset), cls_batch_size)
                x, y_hat = dataset[x_idx]
                # apply model and compute loss
                y = model.forward(x.to(device))
                loss = F.nll_loss(y, y_hat.to(device))
                # update model parameters
                optim.zero_grad()
                loss.backward()
                optim.step()
                # add loss value to list
                train_losses.append(loss.item())

                if (i % cls_eval_interval == 0):
                    # evaluate model
                    model.eval()
                    with torch.no_grad():
                        # apply model to test data
                        y = model.forward(x_test.to(device))
                        loss = F.nll_loss(y, y_test.to(device))
                        # add loss to list
                        test_losses.append(loss.item())
                        # get predictions and compute f1-scores
                        y_pred = y.argmax(-1).cpu().numpy()
                        weighted_f1_scores.append(f1_score(y_test.numpy(), y_pred, average='weighted'))
                        acc_scores.append(accuracy_score(y_test.numpy(), y_pred))
                    # back to training the model
                    model.train()
            # move model back to cpu and
            # set it to evaluation mode
            model.to('cpu').eval()
            # return model and tracked values
            return model, {
                'train-losses': train_losses, 
                'test-losses':  test_losses, 
                'weighted-f1':  weighted_f1_scores, 
                'acc':          acc_scores
            }


        # ## Training Classification models
        # Here we train three models, i.e.
        #  - one trained on only the originally provided (reduced!) dataset
        #  - one trained on the original together with some generated samples
        #  - and one where the generated examples are filtered by the generative classifier

        # ### Baseline

        # In[17]:


        # build dataset of only the original samples
        orig_train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.concatenate(train_per_label, axis=0)),
            torch.LongTensor(np.concatenate([
                (np.zeros(X.shape[0]) + i) for i, X in enumerate(train_per_label)
            ], axis=0))
        )
        # train model on dataset
        model_orig, metrics_orig = train_classifier(orig_train_dataset)
        # plot the losses
        figure(figsize=(10, 5))
        grid()
        plot(metrics_orig['train-losses'], label='train')
        ticks = [i * cls_eval_interval for i in range(len(metrics_orig['test-losses']))]
        plot(ticks, metrics_orig['test-losses'], label='test')
        legend()
        savefig(plot_dir / "classifier_loss_orig.pdf")
        close()


        # ### With generated data

        # In[ ]:


        generated_per_label = [gen.numpy() for gen in generate_examples(n_generated_per_class)]
        # full_train_x = generated_per_label
        # create a combined dataset from the original and generated samples
        full_train_x = [
            np.concatenate(both, axis=0)
            for both in zip(train_per_label, generated_per_label)
        ]
        full_train_y = [
            (np.zeros(X.shape[:1]) + i) 
            for i, X in enumerate(full_train_x)
        ]
        # pack all of this in a dataset
        full_train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.concatenate(full_train_x, axis=0)),
            torch.LongTensor(np.concatenate(full_train_y, axis=0))
        )
        # train model on both original and generated data
        model_gen, metrics_gen = train_classifier(full_train_dataset)
        # plot training and test losses
        figure(figsize=(10, 5))
        grid()
        plot(metrics_gen['train-losses'], label='train')
        ticks = [i * cls_eval_interval for i in range(len(metrics_gen['test-losses']))]
        plot(ticks, metrics_gen['test-losses'], label='test')
        legend()
        savefig(plot_dir / "classifier_loss_full.pdf")
        close()


        # ### With generated data filtered by Generative Classifier

        # In[ ]:


        generated_per_label = generate_examples(n_generated_per_class * 10)

        gc.to(device)
        with torch.no_grad():
            # apply generative classifier
            masks_per_label = [
                gc.forward(x_gen.to(device)).flatten() > 0.5
                for x_gen in generated_per_label
            ]
        gc.to('cpu')

        generated_per_label = [
            x_gen[mask, ...].numpy()[:n_generated_per_class, ...]
            for x_gen, mask in zip(generated_per_label, masks_per_label)
        ]
        print(f"Generated Examples: {[len(X) for X in generated_per_label]}")

        # full_train_x = generated_per_label
        # create a combined dataset from the original and generated samples
        full_train_x = [
            np.concatenate(both, axis=0)
            for both in zip(train_per_label, generated_per_label)
        ]
        # full_train_x = generated_per_label
        full_train_y = [
            (np.zeros(X.shape[:1]) + i) 
            for i, X in enumerate(full_train_x)
        ]
        # pack all of this in a dataset
        full_train_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.concatenate(full_train_x, axis=0)),
            torch.LongTensor(np.concatenate(full_train_y, axis=0))
        )
        # train model on both original and generated data
        model_gen_gc, metrics_gen_gc = train_classifier(full_train_dataset)
        # plot training and test losses
        figure(figsize=(10, 5))
        grid()
        plot(metrics_gen_gc['train-losses'], label='train')
        ticks = [i * cls_eval_interval for i in range(len(metrics_gen_gc['test-losses']))]
        plot(ticks, metrics_gen_gc['test-losses'], label='test')
        legend()
        close()


        # ## Evaluating the Models

        # In[ ]:


        df_results = pd.DataFrame.from_dict({
            "model_seed": [model_seed],
            "data_seed": [data_seed],
            "n_originals_per_class": [n_originals_per_class],
            "n_generated_per_class": [n_generated_per_class],
            "vae_z_dim": [vae_z_dim],
            "vae_beta": [vae_beta],
            "gc_training_steps": [gc_training_steps],
            "gc_batch_size": [gc_batch_size],
            "gc_std": [gc_std],
            "gen_neighbors": [gen_neighbors],
            "gen_alpha": [gen_alpha],
            "gen_std": [gen_std],
            "cls_training_steps": [cls_training_steps],
            "cls_batch_size": [cls_batch_size],
            "cls_eval_interval": [cls_eval_interval],
            "f1-orig": [round(max(metrics_orig["weighted-f1"]), 4)],
            "f1-gen": [round(max(metrics_gen["weighted-f1"]), 4)],
            "f1-gen+gc": [round(max(metrics_gen_gc["weighted-f1"]), 4)],
            "acc-orig": [round(max(metrics_orig["acc"]), 4)],
            "acc-gen": [round(max(metrics_gen["acc"]), 4)],
            "acc-gen+gc": [round(max(metrics_gen_gc["acc"]), 4)],
        })


        # ### Comparison of the accuracy

        # In[ ]:


        # also plot
        figure(tight_layout=True, figsize=(10, 5))
        grid()
        plot(metrics_orig["acc"], label="w/o gen")
        plot(metrics_gen["acc"], label="w/ gen")
        plot(metrics_gen_gc["acc"], label="w/ gen+gc")
        legend()
        savefig(plot_dir / "accuracies_graph.pdf")
        close()
        df_results[["acc-orig", "acc-gen", "acc-gen+gc"]]


        # ### Comparison of the F1-Scores

        # In[ ]:


        # also plot
        figure(tight_layout=True, figsize=(10, 5))
        grid()
        plot(metrics_orig["weighted-f1"], label="w/o gen")
        plot(metrics_gen["weighted-f1"], label="w/ gen")
        plot(metrics_gen_gc["weighted-f1"], label="w/ gen+gc")
        legend()
        savefig(plot_dir / "weighted_f1_graph.pdf")
        close()
        df_results[["f1-orig", "f1-gen", "f1-gen+gc"]]


        # ## Save Results to disk

        # In[ ]:


        # append results to csv file
        results_path = plot_root / f"SINGLE-INTERPOLATION-{data_seed}.csv"
        if results_path.exists():
            df_saved = pd.read_csv(results_path, index_col=0, header=None).T
            df_results = pd.concat([df_saved, df_results])
        df_results.T.to_csv(results_path, header=False)
