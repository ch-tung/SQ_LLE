# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 22:15:38 2022
LLE on S(Q)
@author: CHTUNG
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

import time

#%% load
# load csv files
# training set
# Choose which dataset to use
path = './data/'
if 0:
    X_file = path + 'input_grid_all_GPR80.csv'
    Y_file = path + 'target_grid_all.csv'
else:
    X_file = path + 'input_random_all_GPR80.csv'
    Y_file = path + 'target_random_all.csv'
    
fX = open(X_file, 'r', encoding='utf-8-sig')
sq = np.genfromtxt(fX, delimiter=',')

fY = open(Y_file, 'r', encoding='utf-8-sig')
target = np.genfromtxt(fY, delimiter=',')

eta = target[:,0]
kappa = target[:,1]
Z = target[:,3]
A = target[:,2]
lnZ = np.log(Z)
lnA = np.log(A)

n_sq, n_dim = np.shape(sq)
n_sample = 1500

#%% LLE
'''
0. PCA
'''
tStart0 = time.time()

numpy.random.seed(0)
i_sample = np.random.choice(np.arange(n_sq),n_sample)

X = sq[i_sample,:]-1
X = X.T
U, S, Vh = np.linalg.svd(X)
score = np.matmul(U.T,X)

fig_PCA = plt.figure(figsize=(6, 6))
ax = fig_PCA.add_subplot(projection='3d')
ax.scatter(score[0,:],score[1,:],score[2,:],c=eta[i_sample])
plt.show()

tEnd0 = time.time() 
print("PCA cost %f sec" % (tEnd0 - tStart0))   
    
#%%
'''
1. Find neighbours in X space
'''
tStart1 = time.time()

K = 15
#space partitioning
n_part_dim = 3
n_part = 10
index_part = np.zeros((n_part_dim,n_sample))
for i in range(n_part_dim):
    score_reduced = (score[i,:]-np.min(score[i,:]))/(np.max(score[i,:])-np.min(score[i,:]))
    index_part[i,:] = np.floor(n_part*score_reduced)  
index_part = index_part.T

# list of neighbors
mesh_ext = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1]) # 3D
index_ext = np.array([mesh_ext[0].reshape(27),mesh_ext[1].reshape(27),mesh_ext[2].reshape(27)]).T
index_neig = np.zeros((K,n_sample),'int') 
# the i-th column of index_neig are lists of neighbors of point i
for i in range(n_sample):
    index_part_i = index_part[i]
    index_part_i_ext = index_part_i + index_ext
    count_neig = np.zeros(len(index_part))
    for j in range(len(index_part_i_ext)):
        index_neig_j = ((index_part[:,0] == index_part_i_ext[j,0])*
                          (index_part[:,1] == index_part_i_ext[j,1])*
                          (index_part[:,2] == index_part_i_ext[j,2]))
        count_neig = count_neig + index_neig_j
    count_neig[i] = 0 # exclude Xi
    index_neig_i = np.array([x for x,v in enumerate(count_neig) if v>0])
    Z = X[:,index_neig_i].T -  X[:,i].T # displacement vectors of points in the same subdomain
    ED = np.linalg.norm(Z,axis=1) 
    # pick the points with K-smallest Eucledian distances to point i
    index_neig_i = index_neig_i[np.argpartition(ED,K-1)[:K]] 
    index_neig[:,i] = index_neig_i

tEnd1 = time.time() 
print("step 1 cost %f sec" % (tEnd1 - tStart1))   
#%%
'''
2. Solve for reconstruction weights W 
'''        
tStart2 = time.time()

W = np.zeros((K,n_sample))
tol = 1e-4
for i in range(n_sample):
    Z = X[:,index_neig[:,i]].T -  X[:,i].T
    C = np.matmul(Z,Z.T)
    C = C + np.eye(K)*tol*np.trace(C)
    
    Wi = np.linalg.lstsq(C, np.ones(K), rcond=None)[0]
    Wi = Wi/np.sum(Wi)
    W[:,i] = Wi

tEnd2 = time.time()    
print("step 2 cost %f sec" % (tEnd2 - tStart2))   
    
#%%
'''
3. Compute embedding coordinates Y using weights W
'''     
tStart3 = time.time()

M = np.eye(n_sample)
for i in range(n_sample):
    w = W[:,i]
    j = index_neig[:,i]
    M[i,j] = M[i,j] - w.T
    M[j,i] = M[j,i] - w
    for ij1 in range(K):
        for ij2 in range(K):
            M[j[ij1],j[ij2]] = M[j[ij1],j[ij2]] + np.outer(w,w)[ij1,ij2]

# from scipy import sparse
# Ms = sparse.csr_matrix(M)

# Embedding
# from scipy.sparse.linalg import eigs
# evalues, evects = eigs(Ms,k=3,which='SM')
evalues, evects = np.linalg.eig(M)
i_evalues = np.argsort(evalues)
y = evects[:,i_evalues[0:4]]

tEnd3 = time.time()    
print("step 3 cost %f sec" % (tEnd3 - tStart3))   

#%%
fig_LLE = plt.figure(figsize=(6, 6))
# ax = fig_LLE.add_subplot()
# ax.scatter(y[:,1],y[:,2],c=eta[i_sample])
ax = fig_LLE.add_subplot(projection='3d')
ax.scatter(y[:,1],y[:,2],y[:,3],c=eta[i_sample])
plt.show()
    