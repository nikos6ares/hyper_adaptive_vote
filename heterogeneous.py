# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 16:28:04 2021

@author: Nikos
"""

import numpy as np
import random
import sys
from iteration_utilities import unique_everseen
from random import randrange
import copy
import statistics
import matplotlib.pyplot as plt
from collections.abc import Iterable
from itertools import zip_longest


from multiprocessing import Pool
import multiprocessing
from itertools import repeat
from iterating import  m_rho_iter_traj,iterations
from hypergraph import heterogeneous_initial
from tools import assign_opinions_asymmetry,group_opinions,Nkl_calculator, assign_hyper_opinions, assign_opinions, magnetization_density,moments

import pandas
import os

from plot_heterogeneous import *

def final_abs_magnetization(m_multi_array, rho_multi_array):
    """Calculates the mean and the variance of the absolute final magnetization for a specific parameter"""
    mean_fin_m=0 #mean of final absolute magnetizations
    var_fin_m=0 #variance of final absolute magnetizations
    stdev=0 #sqrt of var
    fin_m=[0]*len(m_multi_array) #final absolute magnetization for every trajectory
    for j in range(len(m_multi_array)):
        mean_fin_m+=abs(m_multi_array[j][-1])/len(m_multi_array)
        fin_m[j]=abs(m_multi_array[j][-1])
    for j in range(len(m_multi_array)):
        var_fin_m+=(abs(m_multi_array[j][-1])-mean_fin_m)**2/len(m_multi_array)
    stdev=np.sqrt(var_fin_m)
    return mean_fin_m, fin_m, stdev

def abs_m_vs_gamma_one_config(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it, num_intervals,p,gamma,alpha):
    """Calculates absolute magnetization versus gamma"""
    gamma_value=np.linspace(0.01,0.49,num_intervals)
    rho_simulations=[0]*num_intervals
    magnetization_simulations=[0]*num_intervals
    
    mean_fin_array=[0]*num_intervals
    fin_m_array=[0]*num_intervals
    stdev_array=[0]*num_intervals
        
    


    for i in range(num_intervals):
        magnetization_simulations[i], rho_simulations[i]=m_rho_iter_traj(num_simulations,hypergraph, op_dict,num_it,gamma_value[i],p,alpha)

        mean_fin_array[i],fin_m_array[i],stdev_array[i]=final_abs_magnetization(magnetization_simulations[i], rho_simulations[i])
    
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)
    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    
    
    dirname = "../Plots/heterogeneous/absm_vs_gamma_N_%d_n_%d_p_%s_alpha_%s_nsim_%d_ninterv_%d" %(N,n,sp,sa,num_simulations,num_intervals)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    df = pandas.DataFrame(data={"gamma_value": gamma_value, "mean_fin_array": mean_fin_array, "stdev_array":stdev_array})
    df.to_csv(dirname+"/absm_vs_gamma_N_%d_n_%d_p_%s_alpha_%s_beta_%s_nsim_%d_ninterv_%d.csv" %(N,n,sp,sa,sb,num_simulations,num_intervals), sep=',',index=False)    
    return gamma_value, mean_fin_array, stdev_array

def m_vs_gamma_multi_config(asymmetry,num_configuration,num_simulations,num_it, num_intervals,p,alpha,beta):
    """Calculates magnetization versus gamma for many trajectories for many configurations"""
    gamma_value=np.linspace(0.18,0.27,num_intervals)
    # beta_value=np.linspace(0.01,1.5,num_intervals)
    
    mean_fin=[0]*num_configuration
    fin_m=[0]*num_configuration
    stdev=[0]*num_configuration
    
    mean_magnet_multi=[0]*num_intervals
    stdev_magnet_multi=[0]*num_intervals
    

    for i in range(num_intervals):
        for j in range(num_configuration):
            # H=heterogeneous_initial(N,n,beta_value[i])
            H=heterogeneous_initial(N,n,beta)

            op_dict=assign_opinions_asymmetry(N,asymmetry)
            
            magnetization_simulations, rho_simulations=m_rho_iter_traj(num_simulations,H, op_dict,num_it,gamma,p,alpha)
            mean_fin[j],fin_m[j],stdev[j]=final_abs_magnetization(magnetization_simulations, rho_simulations)
        
        mean_magnet_multi[i],stdev_magnet_multi[i]=moments(mean_fin)
            
        
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    # sg=str(gamma)
    # sg=[k for k in sg if k!='.']
    # sg="".join(sg)
    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    
    dirname = "../Plots/heterogeneous/multi_absm_vs_gamma_N_%d_beta_%s_n_%d_p_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d" %(N,sb,n,sp,sa,num_simulations,num_configuration,asymmetry)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    # df = pandas.DataFrame(data={"beta_value": beta_value, "mean_magnet_multi": mean_magnet_multi, "stdev_magnet_multi":stdev_magnet_multi})
    df = pandas.DataFrame(data={"gamma_value": gamma_value, "mean_magnet_multi": mean_magnet_multi, "stdev_magnet_multi":stdev_magnet_multi})

    # df.to_csv(dirname+"/multi_absm_vs_beta_N_%d_gamma_%s_n_%d_p_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sg,n,sp,sa,num_simulations,num_configuration,asymmetry), sep=',',index=False)    
    df.to_csv(dirname+"/multi_absm_vs_gamma_N_%d_beta_%s_n_%d_p_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sb,n,sp,sa,num_simulations,num_configuration,asymmetry), sep=',',index=False)    
    return gamma_value, mean_magnet_multi, stdev_magnet_multi


def m_vs_beta_multi_config(asymmetry,num_configuration,num_simulations,num_it, num_intervals,p,alpha,gamma):
    """Calculates magnetization versus beta for many trajectories for many configurations"""
    beta_value=np.linspace(0.01,1.5,num_intervals)
    
    mean_fin=[0]*num_configuration
    fin_m=[0]*num_configuration
    stdev=[0]*num_configuration
    
    mean_magnet_multi=[0]*num_intervals
    stdev_magnet_multi=[0]*num_intervals
    

    for i in range(num_intervals):
        for j in range(num_configuration):
            H=heterogeneous_initial(N,n,beta_value[i])

            op_dict=assign_opinions_asymmetry(N,asymmetry)
            
            magnetization_simulations, rho_simulations=m_rho_iter_traj(num_simulations,H, op_dict,num_it,gamma,p,alpha)
            mean_fin[j],fin_m[j],stdev[j]=final_abs_magnetization(magnetization_simulations, rho_simulations)
        
        mean_magnet_multi[i],stdev_magnet_multi[i]=moments(mean_fin)
            
        
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)

    dirname = "../Plots/heterogeneous/multi_absm_vs_beta_N_%d_gamma_%s_n_%d_p_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d" %(N,sg,n,sp,sa,num_simulations,num_configuration,asymmetry)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    df = pandas.DataFrame(data={"beta_value": beta_value, "mean_magnet_multi": mean_magnet_multi, "stdev_magnet_multi":stdev_magnet_multi})

    df.to_csv(dirname+"/multi_absm_vs_beta_N_%d_gamma_%s_n_%d_p_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sg,n,sp,sa,num_simulations,num_configuration,asymmetry), sep=',',index=False)    
    return beta_value, mean_magnet_multi, stdev_magnet_multi

#------------------------Parameters------------------------------#
N=100 #number of nodes
n=50 #number of hyperedges
beta=0.5 #variance of exponential distribution

gamma=0.05 #threshold for split
p=0.5 #probability of split/merge for 2-edges
alpha=1.0
num_it=50000 #number of iterations 
num_simulations=2 #number of simulations for statistics
num_intervals=2 #number of different gamma values for magnetization vs density function
num_configuration=2
asymmetry=55
#----------------------------------------------------------------#

# # ---calculates/plots abs magnetization versus gamma for 1 configuration----
# H=heterogeneous_initial(N,n,beta)
# op_dict=assign_opinions_asymmetry(N,asymmetry)
# opinions_H=assign_hyper_opinions(op_dict,H)
# gamma_value, mean_fin_array, stdev_array=abs_m_vs_gamma_one_config(num_simulations,H, opinions_H, op_dict,num_it, num_intervals,p,gamma,alpha)
# plot_abs_m_vs_gamma_one_config(gamma_value, mean_fin_array, stdev_array,num_simulations,H, opinions_H, op_dict,num_it, num_intervals,p,gamma,alpha,beta,N,n)
# # ----

# # ---calculates/plots abs magnetization versus gamma for multiple configuration----
# gamma_value, mean_magnet_multi, stdev_magnet_multi=m_vs_gamma_multi_config(asymmetry,num_configuration,num_simulations,num_it, num_intervals,p,alpha,beta)
# plot_m_vs_gamma_multi_config(gamma_value, mean_magnet_multi, stdev_magnet_multi,asymmetry,num_configuration,num_simulations,num_it, num_intervals,p,alpha,beta,N,n)
# # ----

# # ---calculates/plots abs magnetization versus beta for multiple configuration----
#beta_value, mean_magnet_multi, stdev_magnet_multi=m_vs_beta_multi_config(asymmetry,num_configuration,num_simulations,num_it, num_intervals,p,alpha,gamma)
#plot_m_vs_beta_multi_config(beta_value, mean_magnet_multi, stdev_magnet_multi,asymmetry,num_configuration,num_simulations,num_it, num_intervals,p,alpha,gamma,N,n)
# # ----
