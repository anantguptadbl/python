# SGD based Factorization machines

# Configuration of learning Parameters
# This is the learning rate for adjustment to gradients
learningRate=0.00001  
# This is the regParam which determines that the weight with the smallest variance is taken
regParam=0.001
# This is the regParam for updating the biases
biasParam=0.01
# RMSE threshold
rmseThreshold=1e-4

# Input Data
a=np.array([[5,2,0,0,-2],[5,2,2,0,-2],[-2,2,2,0,0]])

# Initialization
x=np.random.rand(3,2)
y=np.random.rand(5,2)
biasx=np.zeros(3).reshape(3,1)
biasy=np.zeros(5).reshape(5,1)
numElements=a.shape[0] * a.shape[1]
loopCounter=0

# Calculation
while(1):
    loopCounter+=1
    error=a-((np.matmul(x,y.T) + biasx) + biasy.T)
    if(np.isnan(error.sum())):
        print("x is {} and y is {} and error is {}".format(x,y,error))
        print("We will have to retry because of SGD failing us")
        break
    if(loopCounter % 100==0):
        print("Error : {} for iteration {}".format(error.sum(),loopCounter))
    if(abs(1.0 * (error.sum()/numElements)) < rmseThreshold):
        print("Achieved")
        break
    else:
        biasx = biasx + (learningRate * (error - biasParam*biasx))
        biasy = biasy + (learningRate * (error.T - biasParam*biasy))
        x=x + ( learningRate * (2 * np.matmul(error,y)) + regParam * x)
        y=y + ( learningRate * (2 * np.matmul(x.T,error)).T + regParam * y)
