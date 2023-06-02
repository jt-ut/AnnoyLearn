import os
import numpy as np
from sklearn import datasets
from AnnoyLearn import VQLearn
import matplotlib.pyplot as plt

# Load iris 
data = datasets.load_iris().data 
labels = datasets.load_iris().target

N = data.shape[0]
d = data.shape[1]

# Generate random starting prototypes in range of data 
M = 10 # number of desired prototypes 
W0 = np.random.uniform(low=data.min(), high=data.max(), size=(M,d))

# Set Neural Gas learning parameters 
n_epochs = 100 
rho0 = np.sqrt(M)
rho_anneal = 0.95 
rho_min = 0.75 
min_h = 0.01 

# Initialize learner class 
learn = VQLearn(d=d, X=data.reshape(-1,1), W=W0.reshape(-1,1), n_epochs=n_epochs, 
                rho0=rho0, rho_anneal=rho_anneal, rho_min=rho_min, min_h=min_h)

# Fit
# This uses OMP in parallel. Can change number of threads used for calculation. 
os.environ["OMP_NUM_THREADS"] = "5"
learn.train()

# Extract learned prototyeps 
W = np.array(learn.W, order='C').reshape(M,d)

# Plot pairwise data + prototypes
plotdim1 = 0; plotdim2 = 3
plt.scatter(x=data[:,plotdim1], y=data[:,plotdim2], c='black', s=2)
plt.scatter(x=W0[:,plotdim1], y=W0[:,plotdim2], c='blue', s=4, marker='s')
plt.scatter(x=W[:,plotdim1], y=W[:,plotdim2], c='red', s=4, marker='s')
plt.show()



