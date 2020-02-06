import argparse
import os
import numpy as np
import math

from torch.autograd import Variable
from torchvision.utils import save_image
import torch

import q2_models

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space") # dodgy?
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--gpu", type=int, default=0, help="GPU number")
parser.add_argument("--gen_name", type=str, default="generator1.pth.tar", help="name of saved generator to load")
parser.add_argument("--output_dir", type=str, default="./output/", help="directory for output images")
parser.add_argument("--save_dir", type=str, default="./log/", help="directory to save trained model")
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

# Load generator
generator = q2_models.Generator(opt)

if cuda:
    generator.cuda()

generator_model_pth = os.path.join(opt.save_dir, "generator_ACGAN.pth.tar")  # hardcode the path once finalised

if os.path.exists(generator_model_pth):
    print("---> found saved model {}, loading checkpoint...".format(opt.gen_name))
    checkpoint = torch.load(generator_model_pth)
    generator.load_state_dict(checkpoint)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

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
print("---> generating images...")
gen_imgs = generator(z, labels)

# Save images
if not os.path.exists(opt.output_dir):
    print("---> created output directory...")
    os.makedirs(opt.output_dir)

print("---> saving images...")
save_image(gen_imgs.data, os.path.join(opt.output_dir, "fig2_2.jpg"), nrow=5, normalize=True)

print("***** Finished Testing *****")