#  Particle Swarm Optimization
import random

def lossFunction(x):
    total=0
    for curDim in range(len(x)):
        total=total+x[curDim]**2
    return(total)

class Particle(object):
    def __init__(self,initData):
        self.numDimensions=len(initData)
        self.positionI=[]
        self.velocityI=[]
        self.posBestI=[]
        self.errBestI=-1
        self.errI=-1
        for curDim in range(self.numDimensions):
            self.velocityI.append(random.uniform(-1,1))
            self.positionI.append(initData[curDim])
        
    def evaluate(self,lossFunction):
        self.errI=lossFunction(self.positionI)
        if(self.errI < self.errBestI or self.errBestI==-1):
            self.posBestI=self.positionI
            self.errBestI=self.errI
            
    def updateVelocity(self,posBestG):
        # Here posBestG means the global dataset
        w=0.5
        c1=1
        c2=1
        
        for curDim in range(self.numDimensions):
            r1=random.random()
            r2=random.random()
            velI=c1*r1*(self.posBestI[curDim]-self.positionI[curDim])
            velG=c2*r2*(posBestG[curDim] - self.positionI[curDim])
            self.velocityI[curDim]=w*self.velocityI[curDim] + velI + velG
            
    def updatePosition(self,bounds):
        for curDim in range(self.numDimensions):
            self.positionI[curDim]=self.positionI[curDim] + self.velocityI[curDim]
            if self.positionI[curDim]>bounds[curDim][1]:
                self.positionI[curDim]=bounds[curDim][1]
            # adjust minimum position if neseccary
            if self.positionI[curDim] < bounds[curDim][0]:
                self.positionI[curDim]=bounds[curDim][0]
        
class ParticleSwarmOptimization():
    def __init__(self,costFunc,initData,bounds,numParticles,maxIter):
        self.errBestG=-1
        self.posBestG=[]
        swarm=[]
        for curParticle in range(numParticles):
            swarm.append(Particle(initData))
        # Iteration
        for curIteration in range(maxIter):
            for curParticle in range(numParticles):
                swarm[curParticle].evaluate(costFunc)
                if(swarm[curParticle].errI < self.errBestG or self.errBestG==-1):
                    self.posBestG=list(swarm[curParticle].positionI)
                    self.errBestG=float(swarm[curParticle].errI)
                    
            for curParticle in range(numParticles):
                swarm[curParticle].updateVelocity(self.posBestG)
                swarm[curParticle].updatePosition(bounds)
                
        print("Finally posBestG {0} and errBest {1}".format(self.posBestG,self.errBestG))

initData=[5,5]               # initial starting location [x1,x2...]
bounds=[(-10,10),(-10,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]
ParticleSwarmOptimization(lossFunction,initData,bounds,numParticles=15,maxIter=30)
        
