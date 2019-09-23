#### ONLY CIRCLES
import numpy as np
import imageio
import os
import matplotlib.pyplot as plt
%matplotlib inline
import cv2

# TRAINING DATA
imageData=np.array([])
imageY=[]
for i,curShape in enumerate(['circle','triangle','star','square']):
    print("Currently executing for {0}".format(curShape))
    for curFile in os.listdir("../input/shapes/{0}".format(curShape)):
        curData=imageio.imread("../input/shapes/{0}/{1}".format(curShape,curFile)).reshape(200,200,1)
        #curData=cv2.resize(curData,(50,50))
        if(imageData.shape[0] ==0):
            imageData=curData
        else:
            imageData=np.append(imageData,curData,axis=0)
        imageY.append(i)
    print("Completed executing for {0}".format(curShape))
imageData=imageData.reshape(-1,1,200,200).astype(np.float32)
imageY=np.array(imageY)        
imageY=np.array(imageY).astype(np.float32)
print("Final imageData shape is {0} and that of imageY is {1}".format(imageData.shape,imageY.shape)) # 200,200

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

class Generator(nn.Module):
    def __init__(self,numClasses,latentDim,numChannels):
        super(Generator, self).__init__()
        self.numClasses=numClasses
        self.latentDim=latentDim
        self.numChannels=numChannels
        self.label_emb = nn.Embedding(self.numClasses,self.latentDim)
        self.init_size = 50 
        #self.init_size = opt.img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(self.latentDim, 16 * self.init_size ** 2))

        self.upsample1=nn.Upsample(scale_factor=2)
        self.conv1=nn.Conv2d(16, 16, 3, stride=1, padding=1)
        self.b1=nn.BatchNorm2d(16, 0.8)
        self.relu1=nn.LeakyReLU(0.2, inplace=True)
        self.upsample2=nn.Upsample(scale_factor=2)
        self.conv2=nn.Conv2d(16, 4, 3, stride=1, padding=1)
        self.b2=nn.BatchNorm2d(4, 0.8)
        self.relu2=nn.LeakyReLU(0.2, inplace=True)
        self.conv3=nn.Conv2d(4, self.numChannels, 3, stride=1, padding=1)
        self.tan1=nn.Tanh()
        #self.conv_blocks = nn.Sequential()

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        #print("Size of gen_input is {0}".format(gen_input.size()))
        out = self.l1(gen_input)
        #print("Size of out is {0}".format(out.size()))
        out = out.view(out.shape[0], 16, self.init_size, self.init_size)
        #print("Size of out input to conv layers is {0}".format(out.size()))
        #img = self.conv_blocks(out)
        img=self.upsample1(out)
        #print("Size of img is {0}".format(img.size()))
        img=self.conv1(img)
        #print("Size of img is {0}".format(img.size()))
        img=self.b1(img)
        #print("Size of img is {0}".format(img.size()))
        img=self.relu1(img)
        #print("Size of img is {0}".format(img.size()))
        img=self.upsample2(img)
        #print("Size of img is {0}".format(img.size()))
        img=self.conv2(img)
        #print("Size of img is {0}".format(img.size()))
        img=self.b2(img)
        #print("Size of img is {0}".format(img.size()))
        img=self.relu2(img)
        #print("Size of img is {0}".format(img.size()))
        img=self.conv3(img)
        #print("Size of img is {0}".format(img.size()))
        img=self.tan1(img)
        #print("Size of img output from gen is {0}".format(img.size()))
        return img

class Discriminator(nn.Module):
    def __init__(self,numClasses,latentDim,numChannels):
        self.numClasses=numClasses
        self.latentDim=latentDim
        self.numChannels=numChannels
        self.init_size = 200
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(self.numChannels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )
        
        #self.conv1=nn.Conv2d(in_filters, out_filters, 3, 2, 1)
        #self.relu1=nn.LeakyReLU(0.2, inplace=True)
        #self.d1=nn.Dropout2d(0.25)
        #self.conv2=nn.Conv2d(in_filters, out_filters, 3, 2, 1)
        #self.relu2=nn.LeakyReLU(0.2, inplace=True)
        #self.d2=nn.Dropout2d(0.25)
        #self.conv3=nn.Conv2d(in_filters, out_filters, 3, 2, 1)
        #self.relu3=nn.LeakyReLU(0.2, inplace=True)
        #self.d3=nn.Dropout2d(0.25)
        #self.conv4=nn.Conv2d(in_filters, out_filters, 3, 2, 1)
        #self.relu4=nn.LeakyReLU(0.2, inplace=True)
        #self.d4=nn.Dropout2d(0.25)

        # The height and width of downsampled image
        ds_size = self.init_size // 2 ** 4

        # Output layers
        #self.adv_layer = nn.Sequential(nn.Linear(16 * ds_size ** 2, 1), nn.Sigmoid())
        #self.aux_layer = nn.Sequential(nn.Linear(16 * ds_size ** 2, self.numClasses), nn.Softmax())
        self.adv_layer = nn.Sequential(nn.Linear(21632, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(21632, self.numClasses), nn.Softmax())

    def forward(self, img):
        #print("Discrim input to conv blocksis {0}".format(img.size()))
        out = self.conv_blocks(img)
        #print("Discrim out conv blocks is {0}".format(out.size()))
        out = out.view(out.shape[0], -1)
        #print("Discrim input adv layer is {0}".format(out.size()))
        validity = self.adv_layer(out)
        #print("Discrim out validity is {0}".format(validity.size()))
        label = self.aux_layer(out)
        #print("Discrim out validity is {0}".format(label.size()))
        return validity, label


# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Optimizers
learningRateG=0.02
learningRateD=0.0001
beta1=0.1
beta2=0.01
latentDim=20
numClasses=4
numChannels=1
batch_size=10
epochs=200

# Initialize generator and discriminator
generator = Generator(numClasses,latentDim,numChannels)
discriminator = Discriminator(numClasses,latentDim,numChannels)


FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor

batch_size=50
learningRateG=0.001
learningRateD=0.00001
#optimizer_G = torch.optim.Adam(generator.parameters(), lr=learningRateG, betas=(beta1, beta2))
optimizer_G = torch.optim.SGD(generator.parameters(), lr=learningRateG, momentum=0.6)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learningRateD)


#numBatches=int(imageData.shape[0]/batch_size)
numBatches=10
generatorExtraCount=4

for epoch in range(epochs):
    totalGLoss=0
    totalDLoss=0
    #for i, (imgs, labels) in range(imageData.shape[0]):
    for curBatch in range(numBatches):
        choiceList=np.random.choice(imageY.shape[0],batch_size)
        #imgs=imageData[curBatch*batch_size:(curBatch+1)*batch_size,:,:,:].astype(np.float32)
        #labels=imageY[numBatches*batch_size:(numBatches+1)*batch_size].astype(np.int)
        imgs=imageData[choiceList,:,:,:].astype(np.float32)
        labels=imageY[choiceList].astype(np.int)
        real_imgs=Variable(torch.from_numpy(imgs))
        labels=Variable(torch.from_numpy(labels))
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)
        for generatorExtraBatch in range(generatorExtraCount):
            # Configure input
            #print("Size of imgs is {0}".format(imgs.shape))
            #real_imgs = Variable(imgs.type(FloatTensor))
            #labels = Variable(labels.type(LongTensor))
            # GENERATOR TRAINING
            # Set the gradients for the Generator
            optimizer_G.zero_grad()
            # Sample noise and random labels as generator input
            z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latentDim))))
            gen_labels = Variable(LongTensor(np.random.randint(0, numClasses, batch_size)))
            # Generate a batch of images
            gen_imgs = generator(z, gen_labels)
            # Loss measures generator's ability to fool the discriminator
            validity, pred_label = discriminator(gen_imgs)
            g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))
            g_loss.backward()
            optimizer_G.step()
        
        # DISCRIMINATOR TRAINING
        optimizer_D.zero_grad()

        # Loss for real images
        real_pred, real_aux = discriminator(real_imgs)
        d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2

        # Loss for fake images
        fake_pred, fake_aux = discriminator(gen_imgs.detach())
        d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        # Calculate discriminator accuracy
        pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()
        totalGLoss=totalGLoss + g_loss.item()
        totalDLoss=totalDLoss + d_loss.item()
    if(epoch %1==0):
        print("[Epoch {0}/{1}] [D loss:{2}] [G loss: {3}]".format(epoch, epochs, d_loss.item(),g_loss.item()))
print("Training Completed")
        
# See what kind of image is getting generated by the generator
z = Variable(FloatTensor(np.random.normal(0, 1, (4, latentDim))))
gen_labels = Variable(LongTensor(np.random.randint(0, numClasses, 4)))
# Generate a batch of images
sampleImage=generator(z, gen_labels)
gen_labels=gen_labels.detach().numpy()
plt.imshow(sampleImage.detach().numpy()[0].reshape(200,200))
plt.title("Label = {0}".format(gen_labels[0]))
plt.show()
plt.imshow(sampleImage.detach().numpy()[0].reshape(200,200))
plt.title("Label = {0}".format(gen_labels[1]))
plt.show()
plt.imshow(sampleImage.detach().numpy()[0].reshape(200,200))
plt.title("Label = {0}".format(gen_labels[2]))
plt.show()
plt.imshow(sampleImage.detach().numpy()[0].reshape(200,200))
plt.title("Label = {0}".format(gen_labels[3]))
plt.show()


