import argparse
import os
import numpy as np
import math

from torch.autograd import Variable
from torchvision.utils import save_image
import torch

import q1_models
import q2_models

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--gpu", type=int, default=0, help="GPU number")
parser.add_argument("--gen_name", type=str, default="generator1.pth.tar", help="name of saved generator to load")
parser.add_argument("--output_dir", type=str, default="./output/", help="directory for output images")
parser.add_argument("--save_dir", type=str, default="./Q1_Q2/saved_models/", help="directory to the saved models from the bash script")
parser.add_argument("--n_classes", type=int, default=2, help="number of attribute classes")
opt = parser.parse_args()

# Set seed
np.random.seed(5)

# Check GPU
if torch.cuda.is_available():
    print("GPU found")
    cuda = True
    torch.cuda.set_device(opt.gpu)
else:
    print("GPU not found")
    cuda = False

# Load generators
generator_gan = q1_models.Generator(opt)
generator_acgan = q2_models.Generator(opt)

if cuda:
    generator_gan.cuda()
    generator_acgan.cuda()

generator_gan_model_pth = os.path.join(opt.save_dir, "generator_GAN.pth.tar")
generator_acgan_model_pth = os.path.join(opt.save_dir, "generator_ACGAN.pth.tar")

if os.path.exists(generator_gan_model_pth):
    print("---> found saved model generator_GAN.pth.tar, loading checkpoint...")
    checkpoint = torch.load(generator_gan_model_pth)
    generator_gan.load_state_dict(checkpoint)
else:
    print("ERROR: could not find GAN model file")

if os.path.exists(generator_acgan_model_pth):
    print("---> found saved model generator_ACGAN.pth.tar, loading checkpoint...")
    checkpoint = torch.load(generator_acgan_model_pth)
    generator_acgan.load_state_dict(checkpoint)
else:
    print("ERROR: could not find ACGAN model file")

# Sample noise as generator input
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


# Generate images for GAN
z = Variable(FloatTensor(np.random.normal(0, 1, (32, opt.latent_dim))))  # 32 indicates number of images to generate

# Generate a batch of images
print("---> generating GAN images...")
gen_imgs = generator_gan(z)

# Save images
if not os.path.exists(opt.output_dir):
    print("---> created output directory...")
    os.makedirs(opt.output_dir)

print("---> saving GAN images...")
save_image(gen_imgs.data[:32], os.path.join(opt.output_dir, "fig1_2.jpg"), nrow=5, normalize=True)


# Generate images for ACGAN

# Create 10 pairs of random vectors for model input (each pair is the same vector)
z = np.random.normal(0, 1, (1, opt.latent_dim))
z = np.vstack([z, z])

for i in range(9):
    random_vec = np.random.normal(0, 1, (1, opt.latent_dim))
    z = np.vstack([z, random_vec])
    z = np.vstack([z, random_vec])

z = Variable(FloatTensor(z)) # tensor, 20 rows x latent_dim cols

# Create labels for each input vector. Alternate 0/1/0/1/... (i.e. not smiling, smiling)
labels = np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1])
labels = Variable(LongTensor(labels))   # tensor, 1 row x 20 cols

# Generate images
print("---> generating ACGAN images...")
gen_imgs = generator_acgan(z, labels)

# Save images
print("---> saving ACGAN images...")
save_image(gen_imgs.data, os.path.join(opt.output_dir, "fig2_2.jpg"), nrow=5, normalize=True)


print("***** FINISHED *****")