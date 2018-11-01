def KSStatistic(data1,data2):

    # First we need to order the array
    data1=np.sort(data1,axis=0)
    data2=np.sort(data2,axis=0)

    KSCum1=[]
    for x in range(len(data2)):
        curVal=data2[x]
        KSCum1.append([curVal,(np.sum(data1[0:x]) * 1.0000 )/len(data1)])

    KSCum2=[]
    for x in range(len(data2)):
        curVal=data2[x]
        KSCum2.append([curVal,(np.sum(data2[0:x]) * 1.0000 )/len(data2)])

    import matplotlib.pyplot as plt
    plt.plot([x[0] for x in KSCum1],[x[1] for x in KSCum1])
    plt.plot([x[0] for x in KSCum2],[x[1] for x in KSCum2])
    plt.show()
    
    return([KSCum1,KSCum2])
    
    # KS Test
import numpy as np
data1=np.random.rand(100,1).flatten()
data2=np.random.choice(data1,50)
KSStatistic(data1,data2)
