# Adam Optimizer

# Sample Data
y=np.array([1,2,3,4,5,6,7,8,9,10]).reshape(10,1)
x=np.array([[2,4,6,8,10,12,14,16,18,20],[3,6,9,12,15,18,21,24,27,30]]).reshape(10,2)

# Weight Initialization
weights=np.random.rand(2).reshape(1,2)

alpha=0.0001
beta1=0.9
beta2=0.999
epsilon = 1.000000 / np.power(10,8)
print("The epsilon value is {0}".format(epsilon))


moment1=np.zeros(x.shape[1]).reshape(2,1)
moment2=np.zeros(x.shape[1]).reshape(2,1)

for iterationCount in range(1000000):
    error= y - np.matmul(x,weights.T)
    gradient = - np.matmul(x.T,error)

    moment1 = (beta1 * moment1) + ( 1 - beta1) * gradient
    moment2 = (beta2 * moment2) + ( 1 - beta2) * np.power(gradient,2)
    
    moment1hat = moment1 / ( 1 - np.power(beta1,iterationCount+1))
    moment2hat = moment2 / ( 1 - np.power(beta2,iterationCount+1))
    weights = weights - ((alpha * moment1hat) / ( np.sqrt(moment2hat) + epsilon )).T
    if(iterationCount % 10000 ==0):
        print("epoch {0} RMSE Error {1}".format(iterationCount,np.sum(np.power(error,2))/x.shape[0]))
    if((np.sum(np.power(error,2))/x.shape[0]) < 1):
        break

