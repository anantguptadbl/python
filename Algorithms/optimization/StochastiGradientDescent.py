# We will be applying simple Stochastic Gradient Descent on WX=Y equation

# Data
x=np.random.rand(100,5)
y=np.random.randint(2,size=(100,1))

# Initialization
w=np.random.rand(5,1)
learningRate=0.001

# Iteration
for epoch in range(1000):
    y_pred=np.matmul(x,w)
    error=y_pred - y
    # The weight adjustment
    adjustment=learningRate * np.matmul(x.T,error)
    w = w - adjustment
    print("Cur Error is {}".format(np.sum(error)))
    if( abs(np.sum(error)* 1.0000 ) / x.shape[0]  < 0.001):
        break
