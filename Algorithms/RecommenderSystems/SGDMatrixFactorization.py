# SGD based Factorization machines
learningRate=0.001
regParam=0.01
a=np.array([[5,2,0,0,-2],[5,2,2,0,-2],[-2,2,2,0,0]])
x=np.random.rand(3,2)
y=np.random.rand(5,2)
biasx=np.zeros(3)
biasy=np.zeros(5)
numElements=a.shape[0] * a.shape[1]

while(1):
    error=a-np.matmul(x,y.T)
    if((1.0 * (error.sum()/numElements)) < 0.5):
        print("Achieved")
        break
    else:
        x=x + ( learningRate * (2 * np.matmul(error,y)) + regParam * x)
        y=y + ( learningRate * (2 * np.matmul(x.T,error)).T + regParam * y)
