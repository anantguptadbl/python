# Initialization

class LogisticRegression():
    def __init__(self,data):
        self.weights=np.random.rand(3).reshape(3,1)
        self.epochs=100
        self.data=data

    def getSigmoid(self,x):
        return 1.000 / (1 + np.exp(-x) )

    def runEpoch_GradientDescent(self):
        prediction=self.getSigmoid(np.matmul(self.data[:,0:3],self.weights))
        loss=prediction - self.data[:,3].reshape(100,1)
        gradient=np.matmul(self.data[:,0:3].T,loss)
        self.weights-=gradient
        print("The current loss is {}".format(loss))

lr=LogisticRegression(data)
lr.runEpoch_GradientDescent()
