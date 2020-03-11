# Deep AutoRegressive Networks

class dan(nn.Module):
  def __init__(self):
    super(dan,self).__init__()
    self.hiddenDim=100
    self.l1=nn.Linear(10,5)
    self.b1=nn.BatchNorm1d(5)
    self.l2=nn.Linear(5,3)
    self.b2=nn.BatchNorm1d(3)
    self.l3=nn.Linear(3,1)

  def forward(self,x):
    x=self.b1(F.relu(self.l1(x)))
    x=self.b2(F.relu(self.l2(x)))
    x=self.l3(x)
    return(x)

model=dan()
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)

data=np.random.randint(10,size=(1000))
labels=np.array([data[x+10] for x in range(989)])
data=np.array([data[x:x+10] for x in range(989)])
epochs=10000
batchSize=32
numBatches=int((1000-11)/batchSize)

for curEpoch in range(epochs):
  totalLoss=0
  for curBatch in range(numBatches):
    model.zero_grad()
    inputData=Variable(torch.from_numpy(data[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.float32)))
    inputLabels=Variable(torch.from_numpy(labels[curBatch*batchSize:(curBatch+1)*batchSize].astype(np.float32)))
    output=model(inputData)
    loss=criterion(output.view(-1),inputLabels.view(-1))
    totalLoss=totalLoss + loss.item()
    loss.backward()
    optimizer.step()
  if(curEpoch%1000==0):
    print("Epoch {0} Loss {1}".format(curEpoch,totalLoss))
