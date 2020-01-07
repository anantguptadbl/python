import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
%matplotlib inline

data=[]
dataLabels=[]
for centerX in range(15,35,1):
    for centerY in range(15,45,1):
        for radius in [2,5,7,10,13]:
            image=np.zeros((50,50))
            image = cv2.circle(image, (centerX,centerY), 10, color=(255,0,0), thickness=-1) 
            data.append(image)
            dataLabels.append([0,radius])
            
for centerX in range(15,35,1):
    for centerY in range(15,35,1):
        for radius in [2,5,7,10,13]:
            image=np.zeros((50,50))
            image = image = cv2.rectangle(image, (centerX-radius,centerY-radius),(centerX,centerY), color=(255,0,0), thickness=-1) 
            data.append(image)
            dataLabels.append([1,radius])
            
data=np.array(data)
dataLabels=np.array(dataLabels)

print("Cell Execution Completed")

# W GAN  : Wassertein GAN
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
Tensor =  torch.FloatTensor

torch.manual_seed(42)

# Classes of Generator and Discriminator
class Generator(nn.Module):
    def __init__(self,latentDim):
        super(Generator, self).__init__()
        self.latentDim=latentDim
        self.label_emb = nn.Embedding(2,self.latentDim)
        # The GAN Generator will be multiple Deconv Layers
        self.dconv1=nn.ConvTranspose2d(1024,512, (5, 5), stride=(2, 2), padding=(2, 2)).cuda()
        self.b1=nn.BatchNorm2d(512)
        self.dconv2=nn.ConvTranspose2d(512,256, (4, 4), stride=(2, 2), padding=(2, 2)).cuda()
        self.b2=nn.BatchNorm2d(256)
        self.dconv3=nn.ConvTranspose2d(256,128, (4,4), stride=(2, 2), padding=(2, 2)).cuda()
        self.b3=nn.BatchNorm2d(128)
        self.dconv4=nn.ConvTranspose2d(128,32, (3,3), stride=(2, 2), padding=(2, 2)).cuda()
        self.b4=nn.BatchNorm2d(32)
        self.dconv5=nn.ConvTranspose2d(32,1, (3,3), stride=(2, 2), padding=(2, 2)).cuda()
        self.l1=nn.Linear(79*79,50*50)
        self.b5=nn.BatchNorm1d(2500)
        #self.conv5=nn.Conv2d(32,8, (3,3), stride=(2, 2), padding=(0,0)).cuda()
        #self.pool5=nn.MaxPool2d(kernel_size=1,stride=1,padding=0).cuda()
        #self.conv6=nn.Conv2d(8,1, (3,3), stride=(1, 1), padding=(0, 0)).cuda()
        #self.pool6=nn.MaxPool2d(kernel_size=2,stride=1,padding=0).cuda()

    def forward(self, noise, externalEmbedding,labels):
        #print("Size of externalEmbedding is {0}".format(externalEmbedding.size()))
        #print("Size of rest is {0}".format(torch.mul(self.label_emb(labels), noise).size()))
        gen_input = torch.cat((torch.mul(self.label_emb(labels), noise) , externalEmbedding),dim=1)
        #print("Size of gen_input is {0}".format(gen_input.size()))
        #gen_input = torch.matmul(externalEmbedding, noise)
        out = F.leaky_relu(self.b1(self.dconv1(gen_input.view(-1,1024,4,4)))).cuda()
        out = F.leaky_relu(self.b2(self.dconv2(out))).cuda()
        out = F.leaky_relu(self.b3(self.dconv3(out))).cuda()
        out = F.leaky_relu(self.b4(self.dconv4(out))).cuda()
        out = F.leaky_relu(self.dconv5(out)).cuda()
        out = self.l1(out.view(-1,79*79))
        out = self.b5(out)
        #out = F.leaky_relu(self.conv5(out))
        #out = self.pool5(out).cuda()
        #out = F.relu(self.conv6(out))
        #out = self.pool6(out).cuda()
        #print("Size of gen output is {0}".format(out.size()))
        #out=self.l1(out.view(-1,32*32))
        #return out.view(-1,1,50,50)
        return(out)

class Discriminator(nn.Module):
    def __init__(self,latentDim):
        super(Discriminator, self).__init__()
        self.latentDim=latentDim
        self.init_size = 200
        self.l1=nn.Linear(2503,50*50)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=5,stride=1,padding=0).cuda()
        self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0).cuda()
        self.conv2 = nn.Conv2d(in_channels=8,out_channels=16,kernel_size=4,stride=1,padding=0).cuda()
        self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0).cuda()
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=0).cuda()
        self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2,padding=0).cuda()
        self.l2=nn.Linear(32*4*4,50).cuda()
        self.adv_layer = nn.Linear(50, 1).cuda()
        self.sigm=nn.Sigmoid()

    def forward(self, input,externalEmbedding):
        input=torch.cat((input.view(-1,50*50*1) , externalEmbedding),dim=1).cuda()
        input=self.l1(input).cuda()
        out=self.conv1(input.view(-1,1,50,50)).cuda()
        out=F.relu(self.pool1(out)).cuda()
        out=self.conv2(out).cuda()
        out=F.relu(self.pool2(out)).cuda()
        out=self.conv3(out).cuda()
        out=F.relu(self.pool3(out)).cuda()
        #print("Discrim output {0}".format(out.size()))
        out=self.l2(out.view(-1,32*4*4)).cuda()
        validity = self.adv_layer(out).cuda()
        validity=self.sigm(validity).cuda()
        return validity
    
print("Cell Execution Completed")

import torch
import numpy as np

# Loss functions
adversarial_loss = torch.nn.BCELoss().cuda()

# Optimizers
beta1=0.1
beta2=0.01
latentDim=1024*4*4 - 3
numClasses=2
epochs=50

# Initialize generator and discriminator
generator = Generator(latentDim).cuda()
discriminator = Discriminator(latentDim).cuda()
learningRateG=0.05
learningRateD=0.05
wganC=0.01
batchSize=256
discriminatorExtraCount=1
generatorExtraCount=1
numBatches=int(data.shape[0]/batchSize)
print("Num batches are {0}".format(numBatches))
generatorExtraCount=1
optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=learningRateG)
optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=learningRateD)
FloatTensor = torch.FloatTensor
LongTensor = torch.LongTensor  

genLossArray=[]
discrimLossArray=[]
for epoch in range(epochs):
    totalGLoss=0
    totalDLoss=0
    for curBatch in range(numBatches):
        inpData=data[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.float32)
        inpLabels=dataLabels[curBatch*batchSize:(curBatch+1)*batchSize,:].astype(np.int)
        real_imgs=Variable(torch.from_numpy(inpData),requires_grad=True).cuda()
        labels=Variable(torch.from_numpy(inpLabels[:,1])).cuda()
        # Adversarial ground truths
        valid = Variable(FloatTensor(batchSize, 1). fill_(1.0), requires_grad=False).cuda()
        fake = Variable(FloatTensor(batchSize, 1).fill_(0.0), requires_grad=False).cuda()
        # GENERATOR TRAINING
        for curCount in range(generatorExtraCount):
            optimizer_G.zero_grad()
            # Sample noise and random labels as generator input
            #z = Variable(FloatTensor(np.random.normal(0, 1, (batchSize, latentDim))))
            z= Variable(FloatTensor(np.random.rand(batchSize,latentDim))).cuda()
            gen_labels = Variable(LongTensor(np.array([0] * batchSize))).cuda()
            # Generate a batch of data
            latentDimDict={'low':np.array([1.,0.,0.]),'mid':np.array([0.,1.,0.]),'high':np.array([0.,0.,1.])}
            inpLabelsModified=Variable(torch.from_numpy(np.array([latentDimDict['low'] if x<=10 else latentDimDict['mid'] if x<=15 else latentDimDict['high'] for x in inpLabels[:,1]]).astype(np.float32))).cuda()
            labelClass=Variable(torch.from_numpy(inpLabels[:,0])).cuda()
            gen_imgs = generator(z, inpLabelsModified,labelClass)
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs.view(-1,1,50,50),inpLabelsModified)
            g_loss = adversarial_loss(validity,valid)
            g_loss.backward()
            totalGLoss=totalGLoss + g_loss.item()
            optimizer_G.step()
        for curCount in range(discriminatorExtraCount):
            optimizer_D.zero_grad()
            real_pred = discriminator(real_imgs,inpLabelsModified)
            d_real_loss = adversarial_loss(real_pred,valid)
            fake_pred = discriminator(gen_imgs.detach(),inpLabelsModified)
            d_fake_loss=adversarial_loss(fake_pred,fake)
            d_loss = (d_fake_loss + d_real_loss)/2
            totalDLoss=totalDLoss + d_loss.item()
            d_loss.backward()
            optimizer_D.step()
            # Clipping as part of WGAN
            #for p in discriminator.parameters():
            #    p.grad.data.clamp_(max=wganC,min=-wganC)
        if(epoch %1==0 and curBatch%9==0):
            print("Epoch {0} Batch {1} [D loss:{2}] [G loss: {3}]".format(epoch, curBatch, totalDLoss,totalGLoss))
    genLossArray.append(totalGLoss)
    discrimLossArray.append(totalDLoss)
       
%matplotlib inline
z= Variable(FloatTensor(np.random.rand(1,latentDim))).cuda()
imgGenerated=generator(z, torch.FloatTensor([[1,1,1]]).cuda(),torch.LongTensor([0]).cuda()).cpu().detach().numpy().reshape(50,50)
print(np.max(imgGenerated))
print(np.min(imgGenerated))
plt.imshow(imgGenerated)
