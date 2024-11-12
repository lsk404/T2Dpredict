import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=N
        self.k=k
        self.samples=samples
        self.newindex=0
       # self.synthetic=np.zeros((self.n_samples*N,self.n_attrs))

    def over_sampling(self):
        N=int(self.N/100)
        self.synthetic = np.zeros((self.n_samples * N, self.n_attrs))
        neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        for i in range(len(self.samples)):
            nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            self._populate(N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=self.samples[nnarray[nn]]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1
if __name__ == '__main__':
    import pandas
    X_train = np.load("X_train.npy")
    y_train = np.load("Y_train.npy")
    arr_train = np.hstack((X_train[2400:],y_train[2400:].reshape(-1,1)))
    s_train=Smote(arr_train,N=300)
    dataset = s_train.over_sampling()
    X_train = np.vstack((X_train[:2400],dataset[:,:-1]))
    Y_train = np.vstack((y_train[:2400].reshape(-1,1),dataset[:,-1].reshape(-1,1)))
    print(X_train.shape)
    print(Y_train.shape)
    np.save("X_train_smote.npy",X_train)
    np.save("Y_train_smote.npy",Y_train)
    
    
    ## ==== test ===
    X_test = np.load("X_test.npy")
    y_test = np.load("Y_test.npy")
    arr_test = np.hstack((X_test[600:],y_test[600:].reshape(-1,1)))
    s_test=Smote(arr_test,N=300)
    dataset = s_test.over_sampling()
    X_test = np.vstack((X_test[:600],dataset[:,:-1]))
    Y_test = np.vstack((y_test[:600].reshape(-1,1),dataset[:,-1].reshape(-1,1)))
    print(X_test.shape)
    print(Y_test.shape)
    np.save("X_test_smote.npy",X_test)
    np.save("Y_test_smote.npy",Y_test)
    