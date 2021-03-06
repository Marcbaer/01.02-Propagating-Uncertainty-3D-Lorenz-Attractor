'''
GP-LSTM regression on Lorenz3D data
'''

from __future__ import print_function
import numpy as np
# Keras
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Model assembling and executing
from kgp.utils.assemble import load_NN_configs, load_GP_configs, assemble
from kgp.utils.experiment import train
# Metrics & losses
from kgp.losses import gen_gp_loss
from kgp.metrics import root_mean_squared_error as RMSE
import pickle

np.random.seed(42)

class Lorenz3D(object):
    def __init__(self,shift=1,test=1,pred_mode=1,epochs=50,batch_size=100,hdim=24,sequence_length=12):
        '''
        Initialize Lorenz 3D Class.
        
        Parameters
        ----------
        shift : Integer
            Number of steps to be predicted into the future.
        pred_mode : Integer
            Mode (Dimension to be predicted). 
        test: Integer
            Test number.
        epochs: Integer
            Number of training epochs.
        batch_size: Integer
            Batch size for training.
        hdim: Integer
            Hidden dimensions of LSTM Layer.
        sequence_length: Integer
            Length of input sequences.            
        '''
        self.shift=shift
        self.test=test
        self.pred_mode=pred_mode
        self.sequence_length=sequence_length
        self.batch_size=batch_size
        self.epochs=epochs
        self.hdim=hdim
        #assign data
        self.load_data_lorenz()
        
        
        
    def load_data_lorenz(self):
        '''
        Load Lorenz Attractor data and split into train, test and validation set.
        Assign data to self.data attribute
        '''
        
        total_length=self.sequence_length+self.shift
        
        data = pickle.load(open("./Data/training_data_Py2_T800.pickle", "rb"))    
        data=data['train_input_sequence']      
        data=data[:1500,:3]
    
        #create sequences with length sequence_length
        result = []
        for index in range(len(data) - total_length):
            
            i=data[index: index + total_length]
            k=i[:self.sequence_length]
            j=np.array(i[total_length-1])
            j=j.reshape(1,3)
            k=np.append(k,j,axis=0)
            result.append(k)
            
        result = np.array(result) 
        
        #reshape (#Timesteps,seq_length,#modes)
        
        result=result.reshape(result.shape[0],result.shape[1],3)
        
        train_end=int(0.8*len(result))
        res_train=result[:train_end]
        res_test=result[train_end:]
        
        #sample_size
        valid=int(0.8*len(res_train))
        Input_data=res_train[:,:self.sequence_length,:]
        Output_data=res_train[:,-1,self.pred_mode-1]
    
        Input_data_test=res_test[:,:self.sequence_length,:]
        Output_data_test=res_test[:,-1,self.pred_mode-1]  
        
        X_train=Input_data[:valid,:,:]
        y_training=Output_data[:valid]
        
        X_test=Input_data_test[:,:]
        y_testing=Output_data_test[:]
        
        X_valid=Input_data[valid:,:,:]
        y_validation=Output_data[valid:] 
        
        #Reshape targets
        
        y_train=y_training.reshape(y_training.shape[0],1)
        y_test=y_testing.reshape(y_testing.shape[0],1)
        y_valid=y_validation.reshape(y_validation.shape[0],1)
        
        
        data = {
            'train': [X_train, y_train],
            'valid': [X_valid, y_valid],
            'test': [X_test, y_test],
        }
        
        # Re-format targets
        for set_name in data:
            y = data[set_name][1]
            y = y.reshape((-1, 1, np.prod(y.shape[1:])))
            data[set_name][1] = [y[:,:,i] for i in range(y.shape[2])]
                      
        self.data=data
    
    def build_train_GPLSTM(self):
        '''
        Define GP-LSTM Architecture and Training.
    
        Returns
        -------
        history : Dictionnary
            Training Information.
        y_test : Numpy Array
            Input Test Data.
        y_pred : Numpy Array
            Predicted output.
        var : Numpy Array
            Predicted Variances.
        rmse_predict : Float
            Training Metrics.
        model : Optimized Model
            Optimized Deep Learning Model after Training.
        data : Dictionnary
            Train, test and validation sets.
        '''   
        data=self.data
        
        # Model & training parameters
        nb_train_samples = data['train'][0].shape[0]
        input_shape = data['train'][0].shape[1:]
        nb_outputs = len(data['train'][1])
        gp_input_shape = (1,)

    
        nn_params = {
            'H_dim': self.hdim,
            'H_activation': 'tanh',
            'dropout': 0.0,
        }
    
        gp_params = {
            'cov': 'SEiso', 
            'hyp_lik': np.log(0.3),
            'hyp_cov': [[4.0], [0.1]],
            
            'opt': {'cg_maxit': 2000,'cg_tol': 1e-4,#'deg':3,
                    'pred_var':-100,
                                    
                    },    
        }
        
        # Retrieve model config
        nn_configs = load_NN_configs(filename='lstm.yaml',
                                     input_shape=input_shape,
                                     output_shape=gp_input_shape,
                                     params=nn_params)
        gp_configs = load_GP_configs(filename='gp.yaml',
                                     nb_outputs=nb_outputs,
                                     batch_size=self.batch_size,
                                     nb_train_samples=nb_train_samples,
                                     params=gp_params)
    
        # Construct & compile the model
        model = assemble('GP-LSTM', [nn_configs['1H'], gp_configs['GP']])
        loss = [gen_gp_loss(gp) for gp in model.output_layers]
        model.compile(optimizer=Adam(1e-5), loss=loss)
    
        # Callbacks
        callbacks = [EarlyStopping(monitor='val_mse', patience=2000)]
    
        # Train the model
        history = train(model, data, callbacks=callbacks, gp_n_iter=5,
                        checkpoint='checkpL3_predmode_'+str(pred_mode)+'_test_'+str(test), checkpoint_monitor='val_mse',
                        epochs=self.epochs, batch_size=self.batch_size, verbose=1)
        
        # Finetune the model
        model.finetune(*data['train'],
                       batch_size=self.batch_size,
                       gp_n_iter=100,
                       verbose=0)
          
        # Test the model
        X_test, y_test = data['test']
        X_train,y_train=data['train']
        
        y_pred,var = model.predict(X_test,return_var=True, X_tr=X_train, Y_tr=y_train,batch_size=self.batch_size)
        var=np.array(var)
        rmse_predict = RMSE(y_test, y_pred)
        print('Test predict RMSE:', rmse_predict)
        print('mean variance:', var.mean())
           
        return history,y_test,y_pred,var,rmse_predict,model,data


if __name__ == '__main__':
    
    #Training and Experiment Parameters
    shift=1
    test=1
    pred_mode=1
    epochs=50
    batch_size=100
    sequence_length=6
    hdim=24
    
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
    training_error=training_error
    
    #save results
    
    res={'data':data,'y_test': y_test,'y_pred':y_pred,'var':var,'valid_error':validation_error,'train_error':training_error,'RMSE':rmse_predict,'std':std}
    pickle.dump(res, open('./Results/results_lorenz3d_predmode'+str(pred_mode)+'_test_'+str(test)+'.p', "wb"))

