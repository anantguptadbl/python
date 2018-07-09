# Read full Data
ratings=pd.read_csv('/var/tmp/data1/ratings.csv')

# Restrict to just rating of 5
ratings=ratings[ratings['rating']==5]

# Train test Split
trainData=ratings.sample(frac=0.005,random_state=42)
testData=ratings[ratings['userId'].isin(trainData['userId'])].dropna()

def to_one_hot(data_point_index, array_size):
    temp = np.zeros(array_size)
    temp[data_point_index] = 1
    return temp

# User Vector
userSize=trainData['userId'].nunique()
userDict={}
for i,x in enumerate(trainData['userId'].unique()):
    userDict[x]=to_one_hot(i,userSize)

# Movie Vector
movieListSize=trainData['movieId'].nunique()
movieDict={}
for i,x in enumerate(trainData['movieId'].unique()):
    movieDict[x]=to_one_hot(i,movieListSize)

# Ready the actual data
x_input=[]
for x in trainData['userId'].values[0:10000]:
    x_input.append(userDict[x])
    
y_output=[]
for x in trainData['movieId'].values[0:10000]:
    y_output.append(movieDict[x])
    
x_input=np.array(x_input)
y_output=np.array(y_output)

# We will now convert the data points into the embeddings
userEncoder=sess.run(ae.encoder_1,feed_dict={ae.x_input:x_input,ae.y_input:y_output})
innerUser=sess.run(ae.inner_1,feed_dict={ae.x_input:x_input,ae.y_input:y_output})
innerMovie=sess.run(ae.inner_2,feed_dict={ae.x_input:x_input,ae.y_input:y_output})
decoder_1_weight=sess.run(ae.decoder_1_weight,feed_dict={ae.x_input:x_input,ae.y_input:y_output})
decoder_1_bias=sess.run(ae.decoder_1_bias,feed_dict={ae.x_input:x_input,ae.y_input:y_output})
movieEncoder=np.add(np.matmul(decoder_1_weight,innerMovie),decoder_1_bias)

# QA
print(userEncoder[0:5])
print(movieEncoder[0:5])
print(movieEncoder.shape)
print(userEncoder.shape)

def centeroidnp(arr):
    length = len(arr)
    centroidArr=[]
    for x in range(len(arr[0])):
        centroidArr.append((1.000 * np.sum([y[x] for y in arr]))/length)
    return centroidArr

# Converting the data to centroids
userData=pd.DataFrame(zip(trainData['userId'].values[0:10000],userEncoder),columns=['userId','userVector'])
userCentroid=[]
for curUser in userData['userId'].unique():
    curData=userData[userData['userId']==curUser]
    dataPoints=[x.tolist() for x in curData['userVector'].values]
    userCentroid.append([centeroidnp(np.array(curData['userVector'].values)),str(curUser)])
userCentroid=pd.DataFrame(userCentroid,columns=['userVector','userId'])

movieData=pd.DataFrame(zip(trainData['movieId'].values[0:10000],movieEncoder),columns=['movieId','movieVector'])
movies=pd.read_csv('/var/tmp/data1/movies.csv')
movieDataMerged=movieData.merge(movies,left_on='movieId',right_on='movieId',how='inner')

movieCentroid=[]
for curMovie in movieDataMerged['movieId'].unique():
    curData=movieDataMerged[movieDataMerged['movieId']==curMovie]
    dataPoints=[x.tolist() for x in curData['movieVector'].values]
    movieCentroid.append([centeroidnp(np.array(curData['movieVector'].values)),str(curMovie),curData['title'][0:1].values[0]])
movieCentroid=pd.DataFrame(movieCentroid,columns=['centroid','movieId','movieName'])

# Find similar users
from sklearn.cluster import KMeans
from sklearn import metrics

# We have found out that 35 is the most optimum cluster number
for numClusters in [35]:
    %time kmeans_model = KMeans(n_clusters=numClusters, random_state=1).fit(userCentroid['userVector'].values.tolist())
    labels = kmeans_model.labels_
    userCentroid['label']=labels
    #print("numClusters {} and Silhputte Score ={}".format(numClusters,metrics.silhouette_score(userCentroid['userVector'].values.tolist(), labels, metric='euclidean')))

# We have found out that 25 is the most optimum cluster number
for numClusters in [25]:
    %time kmeans_model = KMeans(n_clusters=numClusters, random_state=1).fit(movieCentroid['centroid'].values.tolist())
    labels = kmeans_model.labels_
    movieCentroid['label']=labels
    
 
