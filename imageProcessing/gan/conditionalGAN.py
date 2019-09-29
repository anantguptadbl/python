# DC GAN
# First we will create the loss with real data
# Then with fake data and then sum it up and update gradients for Discriminator Loss

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self,latentDim):
        super(Generator, self).__init__()
        self.latentDim=latentDim
        self.init_size = 10000 
        self.label_emb = nn.Embedding(1,self.latentDim)
        # The GAN Generator will be a ANN with 3 layers
        self.l1 = nn.Sequential(nn.Linear(self.latentDim+10, self.init_size))
        self.l2 = nn.Sequential(nn.Linear(self.init_size, 5000))
        self.l3 = nn.Sequential(nn.Linear(5000, 1425))

    def forward(self, noise, labels,yVector):
        gen_input = torch.cat((torch.mul(self.label_emb(labels), noise) , yVector),dim=1)
        out = self.l1(gen_input)
        out = self.l2(out)
        out = self.l3(out)
        return out

class Discriminator(nn.Module):
    def __init__(self,latentDim):
        self.latentDim=latentDim
        self.init_size = 200
        super(Discriminator, self).__init__()
        self.l1=nn.Linear(1425+10,500)
        self.l2=nn.Linear(500,50)
        self.adv_layer = nn.Sequential(nn.Linear(50, 1), nn.Sigmoid())

    def forward(self, input, yVector):
        input=torch.cat((input,yVector),dim=1)
        out=self.l1(input)
        out=self.l2(out)
        validity = self.adv_layer(out)
        return validity
    
print("Cell Execution Completed")

import torch
import numpy as np

# Loss functions
adversarial_loss = torch.nn.BCELoss()

# Optimizers
beta1=0.1
beta2=0.01
latentDim=20
numClasses=2
epochs=1

# Initialize generator and discriminator

generator = Generator(latentDim)
discriminator = Discriminator(latentDim)
learningRateG=0.00001
learningRateD=0.001
numBatches=1
generatorExtraCount=8
optimizer_G = torch.optim.Adam(generator.parameters(), lr=learningRateG)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=learningRateD)

FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor  
curData=np.random.rand(1000,1425)
labelVals=np.random.randint(1,size=1000)
yVectorData=np.random.randint(1,size=(1000,10))
batch_size=curData.shape[0]

for epoch in range(epochs):
    totalGLoss=0
    totalDLoss=0
    for curBatch in range(numBatches):
        choiceList=np.random.choice(curData.shape[0],batch_size)
        inpData=curData[choiceList,:].astype(np.float32)
        labels=labelVals[choiceList].astype(np.int)
        real_imgs=Variable(torch.from_numpy(inpData))
        labels=Variable(torch.from_numpy(labels))
        yVector=Variable(torch.from_numpy(yVectorData[choiceList].astype(np.float32)))
        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # GENERATOR TRAINING
        # Set the gradients for the Generator
        optimizer_G.zero_grad()
        # Sample noise and random labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latentDim))))
        gen_labels = Variable(LongTensor(np.array([0] * batch_size)))
        # Generate a batch of data
        gen_imgs = generator(z, gen_labels,yVector)
        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs,yVector)
        g_loss = adversarial_loss(validity, valid)
        g_loss.backward()
        optimizer_G.step()

        # DISCRIMINATOR TRAINING
        optimizer_D.zero_grad()
        # Loss for real images
        real_pred = discriminator(real_imgs,yVector)
        d_real_loss = adversarial_loss(real_pred, valid)
        # Loss for fake images
        fake_pred = discriminator(gen_imgs.detach(),yVector)
        d_fake_loss = adversarial_loss(fake_pred, fake)
        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2
        # Calculate discriminator accuracy
        #pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
        #gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
        #d_acc = np.mean(np.argmax(pred, axis=1) == gt)

        d_loss.backward()
        optimizer_D.step()
    if(epoch %1==0):
        print("[Epoch {0}/{1}] [D loss:{2}] [G loss: {3}]".format(epoch, epochs, d_loss.item(),g_loss.item()))
