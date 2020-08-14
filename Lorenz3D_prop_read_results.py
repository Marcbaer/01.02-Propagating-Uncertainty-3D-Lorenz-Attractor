#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:58:33 2018

@author: marcbar
"""
import numpy as np
from keras.models import load_model
from kgp.layers import GP
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
import pandas as pd
import matplotlib
from matplotlib import rcParams
from kgp.metrics import root_mean_squared_error as RMSE
#np.__version__


plt.rcParams["figure.figsize"] = [9., 3.]
SMALL_SIZE = 10
matplotlib.rc('axes', titlesize=SMALL_SIZE)
matplotlib.rc('font', size=SMALL_SIZE)
plt.tick_params(labelsize=10)


test=1
pred_mode=1

#3D
results = pickle.load(open('./Results/Propagation_results/res_propagation_test'+str(test)+'_predmode_'+str(pred_mode)+'.p', 'rb'))

K_d=results['K_d']
MEAN_d=results['MEAN_d']
VAR_d=results['VAR_d']
X_hist_d=results['X_hist_d']
W=results['W']
w=results['w']
mean_1=results['mean_1']
var_1=results['var_1']
y=np.array(results['y_test'])[0,:,0]


n_steps=results['n_steps']
n_samples=results['n_samples']



point=results['point']
X=np.array(X_hist_d['hist0'][0])
X_initial=results['X_initial'][0,-1,0]


#plotting

std1=var_1**0.5
std1=np.array(std1)
std1=3.5*std1

#plot predicted means

for i in range(1,n_steps):
    pos=[i]
    violin=plt.violinplot(K_d['K{0}'.format(i)].reshape(len(K_d['K{0}'.format(i)])),pos,showmeans = True)
    for pc in [violin['cbars'],violin['cmins'],violin['cmaxes'],violin['cmeans']]:
        pc.set_edgecolor('#2222ff')
    for pc in violin['bodies']:
        pc.set_facecolor('#2222ff')
                      
plt.grid(False)
x=range(1,n_steps,1)
plt.plot(x,y[point:point+n_steps-1],color='red',marker='o',linestyle='--',label='target',fillstyle='none')
plt.xlabel('#step')
plt.title('Lorenz 3D propagation')
plt.scatter(w,X_initial,color='blue',label='sampled distribution')

plt.legend(loc=2, prop={'size': 8})
plt.savefig('./Figures/Propagation_figures/L3D_Prop_nsamples'+str(n_samples)+'_nsteps'+str(n_steps-1)+'.pdf', bbox_inches = "tight")

plt.show()



'''WALK plot'''

x=[]
y1=[X_initial]
y2=[X_initial]
mean1=[]
for i in range(1,n_steps):
    pos=[i]
    y1.append(min(K_d['K{0}'.format(i)]))
    y2.append(max(K_d['K{0}'.format(i)]))
    x.append(i)
    mean1.append(K_d['K{0}'.format(i)].mean())
    
x1=np.append(0,x)
y1=np.array(y1).reshape(n_steps,)
y2=np.array(y2).reshape(n_steps,)    
plt.plot(x,mean1,label='predicted mean',color='blue',marker='o',markersize=2.5,linestyle='')
plt.plot(x,y[point:point+n_steps-1],label='target',marker='o',color='red',fillstyle='none',markersize=2.5,linestyle='')
plt.fill_between(x1,y1,y2,facecolor='lightgrey',label='confidence bound')
plt.scatter(0,X_initial,color='green',label='initial point',s=8)
plt.xlabel('#step')
plt.title('Lorenz 3D: Sampled distributions vs. true values')
plt.legend(loc=4, prop={'size': 12})
plt.savefig('./Figures/Propagation_figures/L3D_uncertainty'+str(n_steps-1)+'.pdf', bbox_inches = "tight")

rmse=RMSE(y[point:point+n_steps-1],mean1)
