# Logistic Regression Class

class LogisticRegression():
    def __init__(self,X,y):
        self.X=X
        self.y=y
        self.weights=np.random.rand(self.X.shape[1]).reshape(self.X.shape[1],1)
        self.epochs=100

    def getSigmoid(self,x):
        return 1.000 / (1 + np.exp(-x) )

    def runEpoch_GradientDescent(self):
        for curEpoch in range(self.epochs):
            prediction=self.getSigmoid(np.matmul(self.X,self.weights))
            loss=prediction - self.y.reshape(self.y.shape[0],1)
            gradient=np.matmul(self.X.T,loss)
            self.weights-=gradient
            if curEpoch % 10 ==0:
                print("For epoch {} the  log loss is {}".format(curEpoch,-np.array(loss).reshape(self.X.shape[0],).sum()))

if __name__=="__main__":
    X=np.array([[1,2,3],[1,1,0],[4,5,6],[4,5,2]])
    y=np.array([0,1,0,1])
    lr=LogisticRegression(X,y)
    lr.runEpoch_GradientDescent()
