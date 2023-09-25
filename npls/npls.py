import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin

class npls(RegressorMixin,BaseEstimator):
    def  __init__(self, n_components:int=2,a:float=3):
        self.n_components = n_components
        self.a=a
    
    def fit(self, xtrain, ytrain):
        """Fits the model to the data (X, y)
        Parameters
        ----------
        X : ndarray
        y : 1D-array of shape (n_samples, )
            labels associated with each sample"""
        x=xtrain.copy()
        y=ytrain.copy()        
        Tt=np.zeros([x.shape[0],self.n_components])
        mass=np.zeros([y.shape[0]])
        y_copy=ytrain.copy()
        w_k_mass=np.zeros([self.n_components,x.shape[1],1])
        w_i_mass=np.zeros([self.n_components,x.shape[2],1])
        bf_array=[]
        
        mmas=np.zeros([x.shape[0],x.shape[1],x.shape[2]])
        z_pz=np.eye(x.shape[1])
        
        for f in range(0,self.n_components):
            z=np.zeros([x.shape[1],x.shape[2]])
            x_product=np.zeros([x.shape[0],x.shape[1],x.shape[2]])            
            for i in range(0,x.shape[0]):
                x_product[i,:,:]=y[i]*x[i,:,:]
            z=x_product.sum(axis=0)
            Wk, S, WI = np.linalg.svd(z)
            w_k=np.array(Wk[:,0]).reshape(x.shape[1],1)
            w_i=np.array(WI[0,:]).reshape(x.shape[2],1)
            w_k_mass[f,:,:]=w_k
            w_i_mass[f,:,:]=w_i            
            for h in range(0,x.shape[0]):
                Tt[h,f]=np.dot(np.dot(w_i.transpose(),x[h,:,:].transpose()),w_k)
            T=np.array(Tt[:,0:f+1]).reshape(x.shape[0],f+1)
            bf=np.dot((np.dot(np.linalg.inv(np.dot(T,T.transpose())-(((self.a))*np.eye(x.shape[0]))),T)).transpose(),
                        y.reshape([x.shape[0],1]))
            bf_array+=[bf]
            WW=np.kron(w_k,w_i).reshape(x.shape[1],x.shape[2])
            for g in range(0,x.shape[0]):
                mmas[g,:,:]=Tt[g,f]*WW
            x=np.array(x-(mmas)) 
            y=(y-(np.dot(T,bf)).reshape(x.shape[0]))
            mass+=(np.dot(T,bf)).reshape(x.shape[0]).reshape(x.shape[0])
            bf=0
        self.bf_array=bf_array
        self.train_error=(np.square(mass - y_copy)).mean(axis=None)
        self.w_k=w_k_mass
        self.w_i=w_i_mass
        
        return self

    def predict(self, xtest):
        x=xtest.copy()
        Tt=np.zeros([x.shape[0],self.n_components])
        y=np.zeros([x.shape[0]])
        mmas=np.zeros([x.shape[0],x.shape[1],x.shape[2]])
        for f in range(0,self.n_components):
            w_k=self.w_k[f,:,:]
            w_i=self.w_i[f,:,:]
            for h in range(0,x.shape[0]):
                Tt[h,f]=np.dot(np.dot(w_i.transpose(),x[h,:,:].transpose()),w_k)
            T=np.array(Tt[:,0:f+1]).reshape(x.shape[0],f+1)
            WW=np.kron(w_k,w_i).reshape(x.shape[1],x.shape[2])
            for g in range(0,x.shape[0]):
                mmas[g,:,:]=Tt[g,f]*WW
            x=np.array(x-(mmas))
            y=(y+(np.dot(T,self.bf_array[f])).reshape(x.shape[0]))
        return y
            
