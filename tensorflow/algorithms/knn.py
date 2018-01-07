import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

mean1=[10,50]
cov1=[[1,0],[0,40]]
mean2=[10,60]
cov2=[[1,0],[0,20]]
import matplotlib.pyplot as plt
x1,y1=np.random.multivariate_normal(mean1,cov1,2000).T
x2,y2=np.random.multivariate_normal(mean2,cov2,2000).T

Xtr=np.column_stack((x1[0:1500],y1[0:1500])) + np.column_stack((x2[0:1500],y2[0:1500]))
Ytr=[0] * 1501 + [1] * 1501
#Xte=np.column_stack((x1[1500:2000],y1[1500:2000])) + np.column_stack((x2[1500:2000],y2[1500:2000]))
Xte=np.array(list(np.column_stack((x1[1500:2000],y1[1500:2000]))) + list(np.column_stack((x2[1500:2000],y2[1500:2000]))))
Yte=[0] * 499 + [1] * 499
plt.figure(figsize=(20,10))
plt.scatter(x1,y1,label='dist 1',color='red')
plt.scatter(x2,y2,label='dist 2',color='blue')
plt.show()

xtr=tf.placeholder("float",(None,2))
xte=tf.placeholder("float",(2))

#distance=tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))))
distance=tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))),1)
pred=tf.arg_min(distance,0)

accuracy=0

init=tf.global_variables_initializer()

# New Xte
Xte=np.array([10,70])

with tf.Session() as sess:
    sess.run(init)
    #for i in [1,2,100,101,201,202,301,302,501,502,601,602,701,702]:
    for i in [1]:
        #nn_index=sess.run(pred,feed_dict={xtr:Xtr,xte:Xte[i,:]})
        #distance1=sess.run(distance,feed_dict={xtr:Xtr,xte:Xte[i,:]})
        nn_index=sess.run(pred,feed_dict={xtr:Xtr,xte:Xte})
        distance1=sess.run(distance,feed_dict={xtr:Xtr,xte:Xte})
        print(distance1)
        print(nn_index)
        print(np.argmin(distance))
        print(distance[np.argmin(distance)])
        #print("Test", i, "Prediction:", Ytr[nn_index],"True Class:", Yte[i])
        # Calculate accuracy
        #if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
        #    accuracy += 1./len(Xte)
print(accuracy)
