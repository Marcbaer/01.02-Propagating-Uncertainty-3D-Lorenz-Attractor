'''
GP-LSTM regression on Lorenz3D data
'''
from __future__ import print_function
import matlab.engine
import numpy as np
from Lorenz3D_GPLSTM import Lorenz3D
# Metrics & losses
from kgp.metrics import root_mean_squared_error as RMSE

from mpi4py import MPI
import pickle

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

np.random.seed(42)

#define functions
  
if __name__ == '__main__':
    
   #Training and Experiment Parameters
    shift=1
    test=1
    epochs=50
    batch_size=100
    sequence_length=6
    hdim=24
    
    #which mode is predicted (forecasted)
    pred_mode=1+rank
    
    #Build and train model
    Lorenz3D=Lorenz3D(shift,test,pred_mode,epochs,batch_size,hdim,sequence_length=sequence_length)
    history,y_test,y_pred,var,rmse_predict,model,data=Lorenz3D.build_train_GPLSTM()
    
    y_test=np.array(y_test)
    y_pred=np.array(y_pred)
    
    std=[i**0.5 for i in var]
    std=np.array(std)
    std2=2*std
 
    validation_error=history.history['val_mse']
    training_error=history.history['mse']
    training_error=np.array(training_error)
    training_error=training_error#/y_test.shape[0]
    
    res={'data':data,'y_test': y_test,'y_pred':y_pred,'var':var,'valid_error':validation_error,'train_error':training_error,'RMSE':rmse_predict,'std':std}
    #pickle.dump(res, open('./Results/results_lorenz3d_predmode'+str(pred_mode)+'_test_'+str(test)+'.p', "wb"))

#finish training for all pred_modes

global_state = np.zeros((3))
local_state = np.zeros((3))
local_state[rank] = rmse_predict

print('rank:',rank,'mean_var:',var[0,:,0].mean(),'RMSE:',rmse_predict)

comm.Allreduce([local_state, MPI.DOUBLE], [global_state, MPI.DOUBLE], MPI.SUM)
if rank==0:print('global_state:',global_state)


''' Propagate Uncertainty'''
    
X_test=data['test'][0]
X_train=data['train'][0]
y_train=data['train'][1]
y_test=np.array(data['test'][1])

#get initial point we want to propagate
point=1250
X_initial=X_test[point:point+1,:,:]

#concatenated only to avoid dimension error
X1=np.concatenate((X_initial,X_initial),axis=0)

#predict mean&var for initial point
mean_1,var_1 = model.predict(X1,return_var=True, X_tr=X_train, Y_tr=y_train,batch_size=500)

print('var',var_1)

xx=np.array(mean_1)
rmse=RMSE(y_test[0,point,0],mean_1[0][0])
print('rmse:',rmse)


mean_1=np.array(mean_1)[0,1,0]
var_1=np.array(var_1)[0,1,0]

#initialize dictionaries to track history

K_d={}
MEAN_d={}
VAR_d={}
X_hist_d={}

#append initial point to history
X_hist_d["hist{0}".format(0)]=[np.array(X_initial)]
MEAN_d["mean{0}".format(0)]=[mean_1]
VAR_d["var{0}".format(0)]=[var_1]

#number of steps propagating into the future
n_steps=150
# number of sampling points per step
n_samples=1000

for n in range(1,n_steps+1):
    K=[]
    MEAN=[]
    VAR=[]
    X_hist=[]
    for i in range(1,n_samples+1,1):
        
        #sample random integer to get index
        if n==1:
            k=0
        else:
            k=np.random.randint(n_samples)
            
        #get mean, var and history of sampled index
        u=MEAN_d["mean{0}".format(n-1)][k]
        v=VAR_d["var{0}".format(n-1)][k]
        std=v**0.5
        X_old=X_hist_d["hist{0}".format(n-1)][k]            
            
        #sample new point
        k2=np.random.normal(u,std)
        k2=np.array(k2).reshape(1,)
        K.append(k2)
    
        #fill global state with all sampled predictions of every mode
        global_state = np.zeros((3))
        local_state = np.zeros((3))
        local_state[rank] = k2
        comm.Allreduce([local_state, MPI.DOUBLE], [global_state, MPI.DOUBLE], MPI.SUM)
        
        #get new history and save it
        new_Hist=global_state
        new_Hist=new_Hist.reshape(1,1,3)
        X_new=np.concatenate((X_old[:,1:,:],new_Hist),axis=1) 
        
        
        #Use new history to predict new mean and var
        X_=np.concatenate((X_old,X_new),axis=0)    
        mean_2,var_2=model.predict(X_,return_var=True)
        mean_2=np.array(mean_2)[0,1,0]
        var_2=np.array(var_2)[0,1,0]        
        
        
        #append new history to list 
        X_hist.append(X_new)
        MEAN.append(mean_2)
        VAR.append(var_2)        

    
    #append new history to dictionnaries
    K_d["K{0}".format(n)]=np.array(K)
    MEAN_d["mean{0}".format(n)]=np.array(MEAN)
    VAR_d["var{0}".format(n)]=np.array(VAR)
    X_hist_d["hist{0}".format(n)]=np.array(X_hist)    
    print('step done:', n)
 
if rank==0:

    #define X-Axis points        
    W={}       
    for i in range(1,n_steps+1):
        
            W['w{0}'.format(i)]=np.full((n_samples),i)
            
    w=np.full((),0)  
    
    res={'data':data,'X_initial':X_initial,'point':point,'K_d': K_d,'MEAN_d':MEAN_d,'VAR_d':VAR_d,'X_hist_d':X_hist_d,'W':W,'w':w,'mean_1':mean_1,'var_1':var_1,'n_samples':n_samples,'n_steps':n_steps,'y_test':y_test}

    pickle.dump(res, open('./Results/res_propagation_test'+str(test)+'_predmode_'+str(pred_mode)+'.p', "wb"))
    
    print('global_state:',global_state)

