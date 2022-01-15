import os
import numpy as np

import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from PIL import Image

%matplotlib inline

print(torch.cuda.is_available())
torch.version.cuda

# FIRST SHAPE GAN

# TORCH CONFIG
torch.manual_seed(42)

# CONFIG
n_classes=4
embedding_dim=10
latent_dim=10
learning_rate=0.001

# CONDITIONAL GAN
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_conditioned_generator = nn.Sequential(
                nn.Embedding(n_classes, embedding_dim),
                nn.Linear(embedding_dim, 16)
            )
        self.latent = nn.Sequential(
            nn.Linear(latent_dim, 4*4*512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.model = nn.Sequential(
            nn.ConvTranspose2d(513, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512, momentum=0.1,  eps=0.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 128, 4, 2, 1,bias=False),
            nn.BatchNorm2d(128, momentum=0.1,  eps=0.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1,bias=False),
            nn.BatchNorm2d(64, momentum=0.1,  eps=0.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 8, 4, 2, 1,bias=False),
            nn.BatchNorm2d(8, momentum=0.1,  eps=0.8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, inputs):
        noise_vector, label = inputs
        label_output = self.label_conditioned_generator(label)
        label_output = label_output.view(-1, 1, 4, 4)
        latent_output = self.latent(noise_vector)
        latent_output = latent_output.view(-1, 512,4,4)
        concat = torch.cat((latent_output, label_output), dim=1)
        image = self.model(concat)
        #print(image.size())
        return image
        
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
    
        self.label_condition_disc = nn.Sequential(
            nn.Embedding(n_classes, embedding_dim),
            nn.Linear(embedding_dim, 3*128*128)
        )
             
        self.model = nn.Sequential(nn.Conv2d(6, 32, 4, 2, 1, bias=False),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(32, 64, 4, 3, 2, bias=False),
                      nn.BatchNorm2d(64, momentum=0.1,  eps=0.8),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(64, 128, 4, 3,2, bias=False),
                      nn.BatchNorm2d(128, momentum=0.1,  eps=0.8),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.Conv2d(128, 256, 4, 3, 2, bias=False),
                      nn.BatchNorm2d(256, momentum=0.1,  eps=0.8),
                      nn.LeakyReLU(0.2, inplace=True), 
                      nn.Flatten(),
                      nn.Dropout(0.4),
                      nn.Linear(2304, 1),
                      nn.Sigmoid()
                      )

    def forward(self, inputs):
        img, label = inputs
        label_output = self.label_condition_disc(label)
        label_output = label_output.view(-1, 3, 128, 128)
        concat = torch.cat((img, label_output), dim=1)
        output = self.model(concat)
        #print(output.size())
        return output
    
print("Model created")

batch_size=16
train_transform = transforms.Compose([
    
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])
train_dataset = datasets.ImageFolder(root='shapes', transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Models
discriminator = Discriminator().cuda()
generator = Generator().cuda()

# Loss
discriminator_loss = nn.BCELoss()
generator_loss = nn.BCELoss()

# Optimizers
D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate, weight_decay=1e-5)
G_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, weight_decay=1e-5)

print("Loss and Optmizer initialized")

#############
# PLAIN RUN
#############

num_epochs = 100
device='cuda'
learning_rate=0.001

for epoch in range(1, num_epochs+1): 
    for index, (real_images, labels) in enumerate(train_loader):
        if index > 100:
            break
        D_optimizer.zero_grad()
        real_images = real_images.to(device)
        labels = labels.to(device)
        labels = labels.unsqueeze(1).long()
        real_target = Variable(torch.ones(real_images.size(0), 1).to(device))
        fake_target = Variable(torch.zeros(real_images.size(0), 1).to(device))
        D_real_loss = discriminator_loss(discriminator((real_images, labels)), real_target)
        # print(discriminator(real_images))
        #D_real_loss.backward()

        noise_vector = torch.randn(real_images.size(0), latent_dim, device=device)  
        noise_vector = noise_vector.to(device)

        generated_image = generator((noise_vector, labels))
        output = discriminator((generated_image.detach(), labels))
        D_fake_loss = discriminator_loss(output,  fake_target)

        # train with fake
        #D_fake_loss.backward()

        D_total_loss = (D_real_loss + D_fake_loss) / 2

        D_total_loss.backward()
        D_optimizer.step()

        # Train generator with real labels
        G_optimizer.zero_grad()
        G_loss = generator_loss(discriminator((generated_image, labels)), real_target)

        G_loss.backward()
        G_optimizer.step()
    print("Epoch {0} Gen loss {1} Discrim loss {2}".format(epoch, G_loss.item(), D_total_loss.item()))

# SAMPLE IMAGES
for image_label in [0, 1, 2, 3]:
    noise_vector = torch.randn(1, latent_dim, device=device)  
    sample_image = generator((noise_vector, torch.LongTensor([[image_label]]).cuda())).cpu().detach()[0].permute(1, 2, 0)
    PIL_image = Image.fromarray(np.uint8(sample_image.numpy()*255)).convert('RGB')
    PIL_image.save("sample_image_label_{0}.png".format(image_label))
