import daal
from daal.data_management import BlockDescriptor_Float64, readOnly, BlockDescriptor_Float32

def initialize_network(X):
    XRows,XColumns=X.getNumberOfRows(),X.getNumberOfColumns()
    input_neurons=XColumns
    #hidden_neurons=input_neurons+1
    #output_neurons=2
    #n_hidden_layers=hiddenLayers
    
    net=list()
    
    # Configuration
    ENCODER1_SIZE=(XColumns,5)
    ENCODER2_SIZE=(5,3)
    DECODER1_SIZE=(3,5)
    DECODER2_SIZE=(5,XColumns)
    
    # Adding the layers
    encoder1Layer = [ { 'name':'encoder1','weights': np.random.uniform(size=ENCODER1_SIZE[0])} for i in range(ENCODER1_SIZE[1]) ]
    net.append(encoder1Layer)
    encoder2Layer = [ { 'name':'encoder2','weights': np.random.uniform(size=ENCODER2_SIZE[0])} for i in range(ENCODER2_SIZE[1]) ]
    net.append(encoder2Layer)
    decoder1Layer = [ { 'name':'decoder1','weights': np.random.uniform(size=DECODER1_SIZE[0])} for i in range(DECODER1_SIZE[1]) ]
    net.append(decoder1Layer)
    decoder2Layer = [ { 'name':'decoder2','weights': np.random.uniform(size=DECODER2_SIZE[0])} for i in range(DECODER2_SIZE[1]) ]
    net.append(decoder2Layer)    
    return net

def activate_sigmoid(sum):
    return (1/(1+np.exp(-sum)))

def forward_propagation(net,input):
    row=input
    for layer in net:
        prev_input=np.array([])
        for neuron in layer:
            #print("Row is {} and neuron[weights] is {}".format(row,neuron['weights']))
            sum=neuron['weights'].T.dot(row)
            
            result=activate_sigmoid(sum)
            neuron['result']=result
            
            prev_input=np.append(prev_input,[result])
        row =prev_input
    
    return row

def sigmoidDerivative(output):
    return output*(1.0-output)

def back_propagation(net,row,expected):
     for i in reversed(range(len(net))):
            layer=net[i]
            errors=np.array([])
            if i==len(net)-1:
                results=[neuron['result'] for neuron in layer]
                errors = expected-np.array(results) 
            else:
                for j in range(len(layer)):
                    herror=0
                    nextlayer=net[i+1]
                    for neuron in nextlayer:
                        herror+=(neuron['weights'][j]*neuron['delta'])
                    errors=np.append(errors,[herror])
            
            for j in range(len(layer)):
                neuron=layer[j]
                neuron['delta']=errors[j]*sigmoidDerivative(neuron['result'])
                
def updateWeights(net,input,lrate):
    for i in range(len(net)):
        inputs = input
        if i!=0:
            inputs=[neuron['result'] for neuron in net[i-1]]

        for neuron in net[i]:
            for j in range(len(inputs)):
                neuron['weights'][j]+=lrate*neuron['delta']*inputs[j]
                
def training(net, epochs,lrate,X,Y,XRows,XColumns):
    errors=[]
    for epoch in range(epochs):
        sum_error=0
        #for i,row in enumerate(X):
        for i in range(XRows):
            curRow=X.getBlockOfRowsAsDouble(i,i+1)[0]
            outputs=forward_propagation(net,curRow)
            expected=Y.getBlockOfRowsAsDouble(i,i+1)[0]
            sum_error+=sum([(expected[j]-outputs[j])**2 for j in range(XColumns)])
            back_propagation(net,curRow,expected)
            updateWeights(net,curRow,0.05)
        if epoch%100 ==0:
            print('>epoch=%d,error=%.3f'%(epoch,sum_error))
            errors.append(sum_error)
    return errors

# Reconfigure this into an autoencoder
X=np.random.rand(100,10).astype(np.float32)
dX = HomogenNumericTable(X, ntype = np.float32)
net=initialize_network(dX)
errors=training(net,1000, 0.05,dX,dX,100,10)
