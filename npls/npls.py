import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
from signal_noise import signal_noise

class npls(RegressorMixin,BaseEstimator):
    def  __init__(self, excitation_wavelenth:np.ndarray,
                  emission_wavelenth:np.ndarray, n_components:int=2,a:float=3,
                  derivative_rang:list=[],norm_func:list=[],
                  crash_norm_name:str=None,crash_norm_value:float=None):
        self.n_components = n_components
        self.a=a
        self.derivative_rang=derivative_rang
        self.norm_func=norm_func
        self.crash_norm_name=crash_norm_name
        self.crash_norm_value=crash_norm_value
        self.excitation_wavelenth=excitation_wavelenth
        self.emission_wavelenth=emission_wavelenth

    def check_smooth_loadings(self,w_i,excitation_wavelenth,
                              w_k,emission_wavelenth, n_component:int) -> dict[str,dict]:
        # print(w_i.shape,excitation_wavelenth.shape,
        #                       w_k.shape,emission_wavelenth.shape, n_component)
        resp_emission=self.chek_smooth_one_model(w_k,emission_wavelenth)
        resp_excitation=self.chek_smooth_one_model(w_i,excitation_wavelenth)
        if self.crash_norm_name is not None:
            for j in range(len(resp_emission[self.crash_norm_name])):
                if resp_emission[self.crash_norm_name][j]<=self.crash_norm_value:
                    error_message_1=f'Emission {n_component} component is a very noisy.'
                    error_message_2=' May be you can choose another norm.'
                    error_message_3=f' Now you choose {self.crash_norm_name} norm'
                    error_message=error_message_1+error_message_2+error_message_3
                    raise ValueError(error_message)
                if resp_excitation[self.crash_norm_name][j]<=self.crash_norm_value:
                    error_message_1=f'Excitation {n_component} component is a very noisy.'
                    error_message_2=' May be you can choose another norm.'
                    error_message_3=f' Now you choose {self.crash_norm_name} norm'
                    error_message=error_message_1+error_message_2+error_message_3
                    raise ValueError(error_message)
        return {
            "Emission":resp_emission,
            "Excitation":resp_excitation
            }


    def chek_smooth_one_model(self,signal,x) -> dict[str, list]:
        model=signal_noise(derivative_rang=self.derivative_rang,
                                    norm_func=self.norm_func)
        response=model.main(signal=signal,x=x)
        return response

    def fit(self, xtrain, ytrain):
        """Fits the model to the data (X, y)
        Parameters
        ----------
        X : ndarray
        y : 1D-array of shape (n_samples, )
            labels associated with each sample


            """
        x=xtrain.copy()
        y=ytrain.copy()

        if len(self.derivative_rang)>0:
            self.snr_emission=list()
            self.snr_excitation=list()
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

            if len(self.derivative_rang)>0:
                response=self.check_smooth_loadings(w_i=w_i[:,0],excitation_wavelenth=self.excitation_wavelenth,
                                           w_k=w_k[:,0],emission_wavelenth=self.emission_wavelenth,
                                           n_component=f)
                self.snr_emission.append(response['Emission'])
                self.snr_excitation.append(response['Excitation'])

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

