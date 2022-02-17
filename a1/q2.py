import numpy as np
import _pickle as pickle
with open("q2/datasets_1/large_train.pkl", "rb") as f:
	data = pickle.load(f)
import matplotlib.pyplot as plt
plt.scatter(data["X"], data["Y"])
plt.show()

B = np.zeros ((data["X"].shape[0], self.k +1), dtype = np.float)

#calculate B
for i=0:x.shape[0]
 for j=0:k+1
  B[i,j] = x[i]**j

np.matmul (B, self.parameters)

#B^T
B.transpose()

#inverse A
np.linalg.inv(A)

#fit part
#W=[(B^TB)^-1]B^T*Y

#fit L2
#W=[(B^T*B + lambda*I)^-1]B^T*Y
I = np.identity(self.k+1)
--------------------------------------------
row = X.shape[0]
column = self.K + 1
B = np.zeros([row, column])
for i in range(0, row):
    for j in range(0, column):
	B[i,j] = np.power(X[i], j)
prediction_y = np.matmul (B, self.parameters)
return prediction_y
-----------------------------------------
size = range(self.K)
a = list(map(lambda x:self._rbf_2d(X, x), size))
myarray = np.asarray(a)
array = myarray.T[0]
# one coloumn 
one =  np.ones((array.shape[0], 1), dtype=np.float)
B = np.append(one, array, axis=1)
# Multiplying Matrices
y = np.matmul(B, self.parameters)
return y 