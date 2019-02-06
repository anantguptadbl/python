from sklearn.cluster import KMeans
from sklearn.metrics import silhoutte_samples, silhoutte_score

metricData=[]
for curCluster in range(10,200,10):
  kmeans=Kmeans(n_clusters=curCluster,random_state=0).fit(data[featureColumns])
  silhouette_avg=silhouette_score(data[featureColumns],kmeans.labels_)
  SSE=kmeans.inertia_
  metricData.append([curCluster,silhouette_avg,SSE])

# CURVE FITTING USING LOGARITHMS
z=np.polyfit([np.log(x[0]) for x in metricData],[x[2] for x in metricData],1)
slopeVals=[]
for x in range(1,200):
  slopeVals.append([x,(np.log(x) * z[0]) + z[1],(1.000 * z[0])/x])
  
import matplotlib.pyplot as plt
plt.plot([x[0] for x in in slopeVals],[x[1] for x in slopeVals],color='red')
plt.show()
plt.plot([x[0] for x in in slopeVals],[x[2] for x in slopeVals],color='blue')
plt.show()

# Ideal optimized cluster number is
optiK=int(np.array([x[0] for x in slopeVals if x[2] < -0.029 and x[2] > -0.031]).mean())
