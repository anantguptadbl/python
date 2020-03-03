# Soft Decision Tree

class SDT(nn.Module):
  def __init__(self,batchSize,inputDim,outputDim,depth):
    super(SDT,self).__init__()
    self.batchSize=batchSize
    self.inputDim=inputDim
    self.outputDim=outputDim
    self.depth=depth
    self.innerNodeCount=2**depth - 1
    self.leafNodeCount=2**depth
    self.penalty_list = [0.1 * (2 ** (-depth)) for depth in range(0, self.depth)] 
    # Models
    self.innerNodes=nn.Sequential(nn.Linear(self.inputDim,self.innerNodeCount),nn.Sigmoid())
    self.leafNodes=nn.Linear(self.leafNodeCount,self.outputDim)

  def forward(self,inputData):
    # We need to calculate the following
    # p(x), Ql(x), P(x), Penalty, Loss
    # p(x) = This gives you the probability output for each node. There is no cumulative nature to it and it is the sigmoid(Linear) combo
    # P(x) = This gives you the cumulative probability from the root to each node. 
    # 1) Calculate p(x)
    pCalc = self.innerNodes(inputData)
    pCalc = torch.unsqueeze(pCalc, dim=2)
    pCalc=torch.cat((pCalc, 1-pCalc), dim=2)
    # 2) Calculate P(x) and Penalty
    # For this we will have to do the calculation at the depth level
    startIndex=0
    endIndex=1
    PCalc = torch.empty(self.batchSize,1,1).fill_(1.) # we will set the starting point to 1 so that subsequent multiplications of probability works
    penalty = torch.tensor(0.)
    for curDepth in range(self.depth):
      pathProb = pCalc[:, startIndex:endIndex, :]
      penalty= penalty + self.penaltyCalculation(curDepth, PCalc, pathProb)  # extract inner nodes in current layer to calculate regularization term
      print("Before PCalc size is {0}".format(PCalc.size()))
      PCalc = PCalc.view(self.batchSize, -1, 1).repeat(1, 1, 2)
      print("After PCalc size is {0}".format(PCalc.size()))
      PCalc = PCalc * pathProb
      startIndex = endIndex
      endIndex = startIndex + 2 ** (curDepth+1)
    # Leaf Nodes
    PCalc=PCalc.view(self.batchSize, self.leafNodeCount)
    output = self.leafNodes(PCalc)
    return(output,penalty)

  def penaltyCalculation(self, curDepth, PCalc, pathProb):
    print("curDepth is {0}".format(curDepth))
    #print("PCalc is {0}".format(PCalc))
    #print("pathProb is {0}".format(pathProb))
    penalty = torch.tensor(0.)    
    PCalc = PCalc.view(self.batchSize, 2**curDepth)
    print("Size of PCalc is {0}".format(PCalc.size()))
    pathProb = pathProb.view(self.batchSize, 2**(curDepth+1))
    print("Size of pathProb is {0}".format(pathProb.size()))
    for node in range(0, 2**(curDepth+1)):
        alpha = torch.sum(pathProb[:, node]*PCalc[:,node//2], dim=0) / torch.sum(PCalc[:,node//2], dim=0)
        penalty -= self.penalty_list[curDepth] * 0.5 * (torch.log(alpha) + torch.log(1-alpha))
    return penalty

# Configuration
batchSize=8
inputDim=4
outputDim=1
depth=3
numEpochs=2

# Model
model=SDT(batchSize,inputDim,outputDim,depth)
criterion=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)

# Data
inputData=Variable(torch.from_numpy(np.random.rand(8,4).astype(np.float32)))
inputLabels=Variable(torch.from_numpy(np.random.rand(8,1).astype(np.float32)))

for curEpoch in range(numEpochs):
  model.zero_grad()
  modelOutput,penalty=model(inputData)
  loss=criterion(modelOutput,inputLabels) + penalty
  loss.backward()
  optimizer.step()
  if(curEpoch % 100==0):
    print("Loss at epoch {0} is {1}".format(curEpoch,loss.item()))
