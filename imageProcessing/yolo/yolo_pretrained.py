import torch.nn as nn
import torch


class Yolo(nn.Module):
    #def __init__(self, num_classes,
    #             anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892), (9.47112, 4.84053),
    #                      (11.2364, 10.0071)]):
    def __init__(self, num_classes,anchors=[(1.3221, 1.73145), (3.19275, 4.00944), (5.05587, 8.09892)]):
        super(Yolo, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors

        self.stage1_conv1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv3 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv4 = nn.Sequential(nn.Conv2d(128, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv5 = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv6 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv7 = nn.Sequential(nn.Conv2d(256, 128, 1, 1, 0, bias=False), nn.BatchNorm2d(128),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv8 = nn.Sequential(nn.Conv2d(128, 256, 3, 1, 1, bias=False), nn.BatchNorm2d(256),
                                          nn.LeakyReLU(0.1, inplace=True), nn.MaxPool2d(2, 2))
        self.stage1_conv9 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv10 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv11 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv12 = nn.Sequential(nn.Conv2d(512, 256, 1, 1, 0, bias=False), nn.BatchNorm2d(256),
                                           nn.LeakyReLU(0.1, inplace=True))
        self.stage1_conv13 = nn.Sequential(nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512),
                                           nn.LeakyReLU(0.1, inplace=True))

        self.stage2_a_maxpl = nn.MaxPool2d(2, 2)
        self.stage2_a_conv1 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False),
                                            nn.BatchNorm2d(1024), nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv2 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False), nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv3 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv4 = nn.Sequential(nn.Conv2d(1024, 512, 1, 1, 0, bias=False), nn.BatchNorm2d(512),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv5 = nn.Sequential(nn.Conv2d(512, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv6 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))
        self.stage2_a_conv7 = nn.Sequential(nn.Conv2d(1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                            nn.LeakyReLU(0.1, inplace=True))

        self.stage2_b_conv = nn.Sequential(nn.Conv2d(512, 64, 1, 1, 0, bias=False), nn.BatchNorm2d(64),
                                           nn.LeakyReLU(0.1, inplace=True))

        self.stage3_conv1 = nn.Sequential(nn.Conv2d(256 + 1024, 1024, 3, 1, 1, bias=False), nn.BatchNorm2d(1024),
                                          nn.LeakyReLU(0.1, inplace=True))
        self.stage3_conv2 = nn.Conv2d(1024, len(self.anchors) * (5 + num_classes), 1, 1, 0, bias=False)

    def forward(self, input):
        output = self.stage1_conv1(input)
        output = self.stage1_conv2(output)
        output = self.stage1_conv3(output)
        output = self.stage1_conv4(output)
        output = self.stage1_conv5(output)
        output = self.stage1_conv6(output)
        output = self.stage1_conv7(output)
        output = self.stage1_conv8(output)
        output = self.stage1_conv9(output)
        output = self.stage1_conv10(output)
        output = self.stage1_conv11(output)
        output = self.stage1_conv12(output)
        output = self.stage1_conv13(output)

        residual = output

        output_1 = self.stage2_a_maxpl(output)
        output_1 = self.stage2_a_conv1(output_1)
        output_1 = self.stage2_a_conv2(output_1)
        output_1 = self.stage2_a_conv3(output_1)
        output_1 = self.stage2_a_conv4(output_1)
        output_1 = self.stage2_a_conv5(output_1)
        output_1 = self.stage2_a_conv6(output_1)
        output_1 = self.stage2_a_conv7(output_1)

        output_2 = self.stage2_b_conv(residual)
        batch_size, num_channel, height, width = output_2.data.size()
        output_2 = output_2.view(batch_size, int(num_channel / 4), height, 2, width, 2).contiguous()
        output_2 = output_2.permute(0, 3, 5, 1, 2, 4).contiguous()
        output_2 = output_2.view(batch_size, -1, int(height / 2), int(width / 2))

        output = torch.cat((output_1, output_2), 1)
        output = self.stage3_conv1(output)
        output = self.stage3_conv2(output)
        return output

import src
#net = Yolo(20)
net=torch.load("whole_model_trained_yolo_coco")

class YOLOModel(nn.Module):
    def __init__(self):
        super(YOLOModel,self).__init__()
        self.yolo=Yolo(1).cuda()
        self.yolo=torch.load("whole_model_trained_yolo_coco")
        self.l1=nn.Linear(425*9*9,16).cuda()
        self.sigm1=nn.Sigmoid().cuda()
        
    def forward(self,x):
        x=self.yolo(x)
        x=x.view(-1,425*9*9)
        x=self.l1(x).cuda()
        x=self.sigm1(x)
        return(x)
    
# MODEL
model=YOLOModel().cuda()
model.eval()

# FREEZING THE WEIGHTS
for param in model._modules['yolo'].parameters():
    param.requires_grad=False

# LOSSES
criterion1=nn.MSELoss().cuda()
criterion2=nn.MSELoss().cuda()
criterion5=nn.BCELoss().cuda()
criterion6=nn.BCELoss().cuda()
criterion7=nn.BCELoss().cuda()
criterion8=nn.BCELoss().cuda()
    
import time
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import time

torch.save(model,"Model_Small_16Batches")
#model=torch.load("Model_Small_16Batches")
batchSize=4
batches=4
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
startTime=time.time()
for epoch in range(1000):
    totalLoss=0
    for curBatch in range(batches):
        model.zero_grad()
        X=Variable(torch.from_numpy(trainingX[curBatch*batchSize:(curBatch+1)*batchSize].reshape(batchSize,3,300,300).astype(np.float32))).cuda()
        Y=Variable(torch.from_numpy(trainingY[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.float32))).cuda()
        XOut=model(X)
        loss1=criterion1(XOut[:,[2,3,7,8,12,13]],Y[:,[2,3,7,8,12,13]])
        loss2=criterion2(XOut[:,[4,5,9,10,14,15]],Y[:,[4,5,9,10,14,15]])
        loss3=criterion5(XOut[:,0],Y[:,0])
        loss4=criterion6(XOut[:,1],Y[:,1])
        loss5=criterion7(XOut[:,6],Y[:,6])
        loss6=criterion8(XOut[:,11],Y[:,11])
        loss= loss1 + loss2 + loss3 + loss4 + loss5 + loss6
        loss.backward()
        optimizer.step()
        #print([loss1.item(),loss2.item(),loss3.item(),loss4.item()])
        totalLoss=totalLoss + loss.item()
    if(epoch %10==0):
        print("Epoch: {0} Loss is {1} TimeTaken is {2}".format(epoch,totalLoss,time.time()-startTime))
    if((epoch %500==0) & (epoch > 1)):
        print("Epoch: {0} Loss is {1} TimeTaken is {2} Saving".format(epoch,totalLoss,time.time()-startTime))
        torch.save(model,"Model_Small_16Batches")
        
print("Cell Execution Completed")
