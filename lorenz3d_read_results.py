#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 13:39:17 2018

@author: marcbar
"""

import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import rcParams

#read results from workstation:

test=1
shift=1
pred_mode=1

results = pickle.load(open('./Results/results_lorenz3d_predmode'+str(pred_mode)+'_test_'+str(test)+'.p', 'rb'))

y_test=results['y_test']
y_pred=results['y_pred']

var=results['var']
std=[i**0.5 for i in var]
std=np.array(std)
std2=2*std


rmse_predict=results['RMSE']
validation_error=results['valid_error']
training_error=results['train_error']
training_error=np.array(training_error)
training_error=training_error

#plot settings
plt.style.use('default')
SMALL_SIZE = 10
matplotlib.rc('axes', titlesize=SMALL_SIZE)
matplotlib.rc('font', size=SMALL_SIZE)
plt.tick_params(labelsize=10)
rcParams.update({'figure.autolayout': True})

# n-step ahead prediction

size=200
point=10
J=np.arange(0,y_test[0,size:size+size].shape[0],1)

plt.figure(figsize=(4,3))
plt.title('Predicted mean vs. true target, n= '+str(shift)+', RMSE=%1.2f' % rmse_predict)
plt.xlabel("#Test point")

plt.plot(J,y_pred[0,point:point+size,0], color='blue',label='predicted mean',linewidth=1.5)

plt.fill_between(J,y_pred[0,point:point+size,0]+std2[0,point:point+size,0],y_pred[0,point:point+size,0]-std2[0,point:point+size,0],label='95% confidence',
    alpha=0.8, facecolor='lightgrey')
  
plt.plot(y_test[0,point:point+size],label='true',color='red',linestyle='dashed')
plt.legend(loc=2, prop={'size': 8})
plt.savefig('./Figures/Lorenz3D_Predictions_shift_'+str(shift)+'.pdf')
plt.show()


#Training convergence
plt.figure(figsize=(4,3))

plt.xlabel("# Epoch")
plt.ylabel('RMSE')
plt.title('Training convergence for n='+str(shift))
plt.plot(validation_error,label='validation_error')
plt.plot(training_error,label='training_error')
plt.legend(loc=1, prop={'size': 10})
plt.savefig('./Figures/TrainConv_shift'+str(shift)+'_test_'+str(test)+'.pdf')
plt.show()

#plot estimated variance histogramm

k=abs(y_pred)-abs(y_test)
k=k[0,:,0]
i=len(k)
v=np.array(var)  
v=v[0,:,0]

# print('min_var: ',min(var))
# print('max_var:', max(var))  
  
plt.figure(figsize=(4,3))
plt.title('Variance estimates, n='+str(shift))
plt.hist(v,bins=100,label='mean_variance='+str(round(v.mean(),3)))
plt.xlabel('variance')
plt.ylabel('#points')
plt.legend(loc=1, prop={'size': 10})
plt.savefig('./Figures/Var_shift'+str(shift)+'_test_'+str(test)+'.pdf')
plt.show() 
