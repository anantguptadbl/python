# NON NEGTIVE MATRIX FACTORIZATION
class NonNegativeMatrixFactorization():
    def __init__(self,inpArr,latentDimensions):
        self.V=inpArr
        self.latentDimensions=latentDimensions
        self.W=np.random.rand(self.V.shape[0],self.latentDimensions)
        self.H=np.random.rand(self.latentDimensions,self.V.shape[1])
    
    def train(self,epochs):
        self.Hmse=0
        self.Wmse=0
        for epoch in range(epochs):
            self.HNew=self.H * (np.dot(self.W.T,self.V)) / (np.dot(np.dot(self.W.T,self.W),self.H))
            self.WNew=self.W * (np.dot(self.V,self.HNew.T))/(np.dot(np.dot(self.W,self.HNew),self.HNew.T))
            self.HmseNew = (np.square(self.HNew - self.H)).mean(axis=None)
            self.WmseNew = (np.square(self.WNew - self.W)).mean(axis=None)
            if(epoch % 100==0):
                print("H MSE = {} and W MSE = {} after step {}".format(Hmse,Wmse,epoch))
            if((self.HmseNew - self.Hmse==0) | (self.WmseNew - self.Wmse==0)):
                print("We will now end as the MSE for the generated matrices have stabilised")
                break
            self.H=self.HNew
            self.W=self.WNew
            self.Hmse=self.HmseNew
            self.Wmse=self.WmseNew

if __name__=="__main__":
    V=np.array([[0,1,1,0],
            [1,0,1,1],
            [1,0,1,0]])
    nnmf=NonNegativeMatrixFactorization(V,2)
    nnmf.train(10000)
    
    # Similar Products
    print("Product Similarity Matrix is ")
    print("{} ".format(np.round(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(nnmf.H.T)),3)))
    
    # Similar Users
    print("USer Similarity Matrix is")
    print(" {}".format(np.round(scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(W)),3)))
