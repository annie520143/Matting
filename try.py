import numpy as np
import numpy.ma as ma

a1 = np.identity(3)

a2 = np.array([[1,0,0]
            ,[0,1,0]
            ,[0,0,1]])
a3 = np.array([[3,1,3]
            ,[2,3,1]
            ,[1,1,1]])
a4 = np.identity(3)
print(a1+a2)

print(np.linalg.inv(a2))

A = np.array([[a1, a2],[a3, a4]])
A.flatten()
b1 = np.array([[1],[2],[1]])
b2 = np.array([[3],[1],[3]])
b = np.array([[b1],[b2]])

print(A.shape)
print(A)
print(b.shape)
print(b)

l = np.array([1,2,3,4,5,6])
print(l[:3])
print(l[2:3])

b = np.array([[1], [11], [-2], [1], [1], [1]])
c = np.array([1,2,3,4,5,6])
fg_reduced = ma.masked_array(c, mask = [1,0,0,0,1,0])

print(fg_reduced.shape)

g = []
g.append([1,2])
g = np.array(g)
print(g.shape)

