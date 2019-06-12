# Generating sample data for our training

# CASE 1
# This is a simple repeated data set
data=np.array([0,1,0,-1,-1,0,1,0,-1,-1,0,1,0,-1,-1,0,1,0,-1,-1])
data=np.array(list(data)*100)
data.shape
data=data[0:1500]

import torch
import torch.nn as nn
import torch.optim as optim


class LSTM(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.encoder = nn.LSTM(self.input_dim, self.latent_dim, self.num_layers)
        self.decoder = nn.LSTM(self.latent_dim, self.input_dim, self.num_layers)

    def forward(self, input):
        # Encode
        _, (last_hidden, _) = self.encoder(input)
        encoded = last_hidden.repeat(input.size()[0],1,1)
        # Decode
        y, _ = self.decoder(encoded)
        # Linear Model
        #print(y.size())
        return torch.squeeze(y)

# LSTM Configuration
totalElements=1500  # This denotes the number of rows. Each row consits of the number of inputElements
batch_size=1       # This is the number of rows that will be used for gradient update
step_size=5         # This is the number of LSTM cells
totalBatches=int(math.ceil(totalElements/(step_size*batch_size)))
inputDim=1
epochLength=3
hiddenDim=2

# LSTM Model
model = LSTM(input_dim=1, latent_dim=hiddenDim, num_layers=1)
#loss_function = nn.BCELoss()
loss_function = nn.MSELoss()
#optimizer = optim.Adam(model.parameters(),lr=0.00001,amsgrad=True)
optimizer = optim.SGD(model.parameters(),lr=0.00001)
print("This cell execution has completed")

for curEpoch in range(epochLength):
    totalLoss=0
    #for curBatch in range(totalBatches):
    for curBatch in range(1):
        y = torch.Tensor(data[curBatch*batch_size*step_size:(curBatch +1)*batch_size*step_size]).view(step_size,-1,inputDim)
        y_pred = model(y)
        optimizer.zero_grad()
        loss = loss_function(y_pred.view(step_size,batch_size,inputDim), y.view(step_size,batch_size,inputDim))
        loss.backward()
        optimizer.step()
        totalLoss=totalLoss + loss
    if(curEpoch % 1==0):
        print("Total loss for {0} is {1}".format(curEpoch,totalLoss))
        paramNames=list(model.named_parameters())
        weights=paramNames[0][1].detach().numpy().reshape(4,hiddenDim,inputDim)
        inputGateWeight=weights[0]
        forgetGateWeight=weights[1]
        stateAdditionGateWeight=weights[2]
        outputGateWeight=weights[3]
        hiddenWeights=paramNames[1][1].detach().numpy().reshape(4,hiddenDim,hiddenDim)
        inputGateWeightHidden=hiddenWeights[0]
        forgetGateWeightHidden=hiddenWeights[1]
        stateAdditionGateWeightHidden=hiddenWeights[2]
        outputGateWeightHidden=hiddenWeights[3]
        print("Input = {0}".format(y))
        #Similary we can get for BIAS as well
        # We will first analyse the movement of weights
        

'W_ii|W_if|W_ig|W_io'
'W_ii = Input Gate'
'W_if = Forget Gate'
'W_ig = Values that will get added to the state'
'W_io = Output Gate'
