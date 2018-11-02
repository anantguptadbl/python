import numpy as np

a=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15],[16,17,18]])
U4,S4,V4=np.linalg.svd(a,full_matrices=True)
S4=np.diag(S4)
for x in range(a.shape[0] - S4.shape[0]):
    S4=np.vstack((S4,np.zeros(3)))

for x in range(a.shape[1] - S4.shape[1]):
    S4=np.hstack((S4,np.zeros(S4.shape[0]).T.reshape(S4.shape[0],1)))
    
print(U4.shape)
print(S4.shape)
print(V4.shape)
np.dot(np.dot(U4,S4),V4)
