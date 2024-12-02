# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 10:40:14 2021

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
from hypergraph import create_hypergraph
from tools import assign_opinions_asymmetry,group_opinions,Nkl_calculator, assign_hyper_opinions, assign_opinions, magnetization_density,moments

from plot_suniform import *
import pandas
import os

def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict



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

def abs_m_vs_gamma(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it, num_intervals,p,gamma,alpha,N,S,n,asymmetry):
    """Calculates average and stdev of absolute final magnetization over many simulations 
    for the same initial configuration for varying gamma"""
    gamma_value=np.linspace(0.01,0.49,num_intervals)
    rho_simulations=[0]*num_intervals
    magnetization_simulations=[0]*num_intervals
    
    mean_fin_array=[0]*num_intervals
    fin_m_array=[0]*num_intervals
    stdev_array=[0]*num_intervals
    
    gamma_array=[0]*num_intervals*num_simulations
    

    for i in range(num_intervals):
        for j in range(num_simulations):
            gamma_array[i*num_simulations+j]=gamma_value[i]
    for i in range(num_intervals):
        magnetization_simulations[i], rho_simulations[i]=m_rho_iter_traj(num_simulations,hypergraph, op_dict,num_it,gamma_value[i],p,alpha)

        mean_fin_array[i],fin_m_array[i],stdev_array[i]=final_abs_magnetization(magnetization_simulations[i], rho_simulations[i])
        
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    dirname = "../Plots/s_uniform/abs_m_vs_gamma_p_%s_N_%d_S_%d_n_%d_asym_%d_alpha_%s_nsim_%d" %(sp,N,S,n,asymmetry,sa,num_simulations)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    df = pandas.DataFrame(data={"gamma_value":gamma_value,"mean_fin_array":mean_fin_array,"fin_m_array":fin_m_array,"stdev_array":stdev_array})
    df.to_csv(dirname+"/abs_m_vs_gamma_p_%s_N_%d_S_%d_n_%d_asym_%d_alpha_%s_nsim_%d.csv" %(sp,N,S,n,asymmetry,sa,num_simulations), sep=',',index=False)

    return mean_fin_array,fin_m_array,stdev_array,gamma_value



def symmetry_cap(num_simulations,num_it, num_config, num_N,size_N,initial_N, mean_degree,S,n,p,alpha,asymmetry):
    """Calculates the asymmetry for different sizes by keeping the mean degree constant by changing S or N
    For every N,n,S it calculates the asymmetry for gamma=0.49 (maximum asymmetry) == cap
    Output: Diagram asymmetry vs N """
    N_array=np.linspace(initial_N, initial_N+(size_N*num_N),num_N,endpoint=False)
    noise_cap=np.zeros(num_N)
    noise_cap_stdev=np.zeros(num_N)

    for k in range(num_N):

        fnoise_array=np.zeros(num_config)
        N=int(N_array[k])

        for i in range(num_config):
            
            H=create_hypergraph(N,S,n)
            op_dict=assign_opinions_asymmetry(N,asymmetry)
            opinions_H=assign_hyper_opinions(op_dict,H)
            
            m_init,rho_init=magnetization_density(opinions_H) #magnetization of initial hypergraph

            magnetization_simulations, rho_simulations=m_rho_iter_traj(num_simulations,H, op_dict,num_it,0.499,p,alpha)
            n_antisym=0
            for j in range(num_simulations):
                if magnetization_simulations[j][-1]<m_init:
                    
                    n_antisym+=1

            fnoise_array[i]=n_antisym/num_simulations
        noise_cap[k], noise_cap_stdev[k]=moments(fnoise_array)
                 
    
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)
        
    dirname = "../Plots/s_uniform/symmetry_restoration_meandeg_%d_p_%s_S_%d_n_%d_asym_%d_gamma_%s_alpha_%s_nsim_%d" %(mean_degree,sp,S,n,asymmetry,sg,sa,num_simulations)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
   
    df = pandas.DataFrame(data={"N_array": N_array, "noise_cap": noise_cap, "noise_cap_stdev":noise_cap_stdev})
    df.to_csv(dirname+"/symmetry_restoration_meandeg_%d_p_%s_S_%d_n_%d_asym_%d_gamma_%s_alpha_%s_nsim_%d.csv" %(mean_degree,sp,S,n,asymmetry,sg,sa,num_simulations), sep=',',index=False)    

    return N_array, noise_cap, noise_cap_stdev
    
def nkl_histogram(k):
    """Creates histogram with N(k,S) for S-Uniform hypergraph. Used to test theorem of bands"""
    bar_width = 0.5 # set this to whatever you want
    data = k
    positions = np.arange(len(k))
    plt.bar(positions, data, bar_width)
    plt.xticks(positions , positions)
    plt.ylabel('Frequency')
    plt.xlabel('Number of nodes with opinion 1 (k)')
    plt.show()
    return

#------------------------Parameters------------------------------#

N=100 #number of nodes
S=10 #size of hyperedges
n=20 #number of hyperedges

# mean_degree=3
# n=int(N*mean_degree/S)
gamma=0.3 #threshold for split
p=0.5 #probability of split/merge for 2-edges
alpha=1.0
num_it=2000 #number of iterations 
num_simulations=100 #number of simulations for statistics
num_intervals=50 #number of different gamma values for magnetization vs density function
num_configuration=50
num_N=3
initial_N=100
size_N=10
asymmetry=50
if S*n<N:
    print("S*n<N: Error: Every node must be in at least one hyperedge of size S")
    sys.exit()
    
#----------------------------------------------------------------#
#-----Creates initial hypergraph
H=create_hypergraph(N,S,n)
op_dict=assign_opinions_asymmetry(N,asymmetry)
opinions_H=assign_hyper_opinions(op_dict,H)
k=Nkl_calculator(H,opinions_H)
nkl_histogram(k)

# # ---Plots histogram N(k,S,0) wrt to k----
# k=Nkl_calculator(H,opinions_H)
# nkl_histogram(k)
# # ----

# # ---Plots histogram magnetization vs density----
# magnetization_vs_density(num_simulations,H, opinions_H, op_dict,num_it, num_intervals,alpha,N,S,n,asymmetry,p)
# # ----

# # ---Plots/calculates absolute magnetization vs gamma----
mean_fin_array,fin_m_array,stdev_array,gamma_value=abs_m_vs_gamma(num_simulations,H, opinions_H, op_dict,num_it, num_intervals,p,gamma,alpha,N,S,n,asymmetry)
# data=pandas.read_csv("absm_vs_gamma_N_100_S_10_n_20_p_05_alpha_10_nsim_301.csv")
# gamma_value=data['gamma_value'].values
# mean_fin_array=data['mean_fin_array'].values
# stdev_array=data['stdev_array'].values
plot_abs_m_vs_gamma(mean_fin_array,stdev_array,gamma_value,num_simulations,num_it, num_intervals,p,alpha,N,S,n,asymmetry)
# # ----

# # ---Plots/calculates symmetry restoration diagram----
# N_array, noise_cap, noise_cap_stdev=symmetry_cap(num_simulations,num_it, num_configuration, num_N,size_N,initial_N, mean_degree,S,n,p,alpha,asymmetry)
# plot_symmetry_cap(N_array, noise_cap, noise_cap_stdev,mean_degree,p,S,n,asymmetry,gamma,alpha,num_simulations)
# # ----

# # ---Compares analytical position of bands with simulations (Loaded data in def plot_bands_on----
# plot_bands_on()
# # ----
