# Adapted from https://github.com/eriklindernoren/PyTorch-GAN#deep-convolutional-gan

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

from tensorboardX import SummaryWriter
import q1_data

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--val_epoch", type=int, default=5, help="on which epoch to save model and image")
parser.add_argument("--batch_size", type=int, default=40, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between image sampling")
parser.add_argument("--gpu", type=int, default=0, help="GPU number")
parser.add_argument("--continue_epoch", type=int, default=1, help="Epoch number which was reached last time. Used to name saved models/images")
parser.add_argument("--gen_name", type=str, default="generator1.pth.tar", help="name of saved generator to load")
parser.add_argument("--disc_name", type=str, default="discriminator1.pth.tar", help="name of saved discriminator to load")
parser.add_argument("--data_dir", type=str, default="./data/", help="root path to the training images")
parser.add_argument("--output_dir", type=str, default="./output/", help="directory for output images")
parser.add_argument("--save_dir", type=str, default="./log/", help="directory to save trained model")
opt = parser.parse_args()

# Directory for output images
if not os.path.exists(opt.output_dir):
    print("Created output directory")
    os.makedirs(opt.output_dir)

# Directory for saved models
if not os.path.exists(opt.save_dir):
    print("Created save directory")
    os.makedirs(opt.save_dir)

if torch.cuda.is_available():
    print("GPU found")
    cuda = True
    torch.cuda.set_device(opt.gpu)
else:
    print("GPU not found")
    cuda = False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, opt.channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(opt.channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = opt.img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity


# Loss function
print("---> preparing loss function...")
adversarial_loss = torch.nn.BCELoss()


# Initialize generator and discriminator
print("---> preparing models...")
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

generator_model_pth = os.path.join(opt.save_dir, opt.gen_name)
discriminator_model_pth = os.path.join(opt.save_dir, opt.disc_name)

if os.path.exists(generator_model_pth):
    print("---> found previously saved {}, loading checkpoint...".format(opt.gen_name))
    checkpoint = torch.load(generator_model_pth)
    generator.load_state_dict(checkpoint)

if os.path.exists(discriminator_model_pth):
    print("---> found previously saved {}, loading checkpoint...".format(opt.disc_name))
    checkpoint = torch.load(discriminator_model_pth)
    discriminator.load_state_dict(checkpoint)


# Initialize weights
print("---> preparing weights...")
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
print("---> preparing dataloader...")
dataloader = torch.utils.data.DataLoader(
    q1_data.DATA(opt),
    batch_size=opt.batch_size,
    num_workers=opt.n_cpu,
    shuffle=True,
)

# Optimizers
print("---> preparing optimizer...")
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# Tensorboard
writer = SummaryWriter(os.path.join(opt.save_dir, "train_info"))

# ----------
#  Training
# ----------

print("---> start training...")
iters = 1
best_loss_G = 1000
best_epoch_G = 0
best_loss_D = 1000
best_epoch_D = 0
for epoch in range(1, opt.n_epochs+1):
    
    #model.train()?

    for i, imgs in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        # Tensorboard
        writer.add_scalar('loss vs iters', d_loss.item(), iters)    # discriminator loss vs num of iterations
        writer.add_scalar('loss vs iters', g_loss.item(), iters)    # generator loss vs num of iterations

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        iters += 1
    
    if epoch % opt.val_epoch == 0:     
        print("")
        print("Saving models/images (current: epoch %d) (total: epoch %d)" % (epoch, opt.continue_epoch))
        print("")

        torch.save(generator.state_dict(), os.path.join(opt.save_dir, "generator{}.pth.tar".format(opt.continue_epoch)))
        torch.save(discriminator.state_dict(), os.path.join(opt.save_dir, "discriminator{}.pth.tar".format(opt.continue_epoch)))
        save_image(gen_imgs.data[:25], os.path.join(opt.output_dir, "img{}.png".format(opt.continue_epoch)), nrow=5, normalize=True)
        opt.continue_epoch += opt.val_epoch


print("")
print("***** FINISHED TRAINING *****")
            

        