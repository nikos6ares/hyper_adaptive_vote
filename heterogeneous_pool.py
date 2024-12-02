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
from iterating import  m_rho_iter_traj_constant,m_rho_iter_traj,iterations
from hypergraph import heterogeneous_initial
from tools import assign_opinions_asymmetry,group_opinions,Nkl_calculator, assign_hyper_opinions, assign_opinions, magnetization_density,moments

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

def abs_m_vs_gamma(num_simulations,hypergraph,N,asymmetry,beta,num_it, num_intervals,p,gamma,alpha):
    """Plots magnetization versus density for a given initial network and for different values of gamma 
    Graphs that can be plotted:
       Mean final magnetization versus gamma with its error bars (characteristic phase transition)
       Magnetization vs Rho (observable phase transition)
       Histograms of normalized magnetizations vs gamma (in order to verify bimodal presence)
       Variance of y coords of m vs binned x (possible secondary phase transiiton) """
    gamma_value=np.linspace(0.01,0.49,num_intervals)
    rho_simulations=[0]*num_intervals
    magnetization_simulations=[0]*num_intervals
    cmap = plt.cm.get_cmap('hsv', num_intervals+1)
    
    mean_fin_array=[0]*num_intervals
    fin_m_array=[0]*num_intervals
    stdev_array=[0]*num_intervals
    

    
    fnoise_array=[0]*num_intervals

        
    gamma_array=[0]*num_intervals*num_simulations
    
    p_value=np.linspace(0.05,1.0,num_intervals)
    p_array=[0]*num_intervals*num_simulations
    #for i in range(num_intervals):
    #    for j in range(num_simulations):
    #
    #        gamma_array[i*num_simulations+j]=gamma_value[i]
    #        p_array[i*num_simulations+j]=p_value[i]
    
    pool = Pool(20)
    a=pool.starmap(m_rho_iter_traj_constant, zip(repeat(num_simulations), repeat(hypergraph), repeat(N),repeat(asymmetry),repeat(num_it),gamma_value,repeat(p),repeat(alpha)))
    magnetization_simulations=[item[0] for item in a]
    rho_simulations=[item[1] for item in a]

    for i in range(num_intervals):
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
            
    dirname = "../heterogeneous/het_absm_vs_gamma_N_%d_n_%d_p_%s_alpha_%s_beta_%s_nsim_%d_asymm_%d" %(N,n,sp,sa,sb,num_simulations,asymmetry)
   
    df = pandas.DataFrame(data={"gamma_value": gamma_value, "mean_fin_array": mean_fin_array, "stdev_array":stdev_array})
    df.to_csv(dirname+"/het_absm_vs_gamma_N_%d_n_%d_p_%s_alpha_%s_beta_%s_nsim_%d_asymm_%d.csv" %(N,n,sp,sa,sb,num_simulations,asymmetry), sep=',',index=False)    

    return 

def multi_config(num_configuration,N,n,beta,asymmetry,num_simulations,num_it,gamma,p,alpha):
    mean_fin=[0]*num_configuration
    fin_m=[0]*num_configuration
    stdev=[0]*num_configuration

    for j in range(num_configuration):
        # H=heterogeneous_initial(N,n,beta_value[i])
        H=heterogeneous_initial(N,n,beta)

        op_dict=assign_opinions_asymmetry(N,asymmetry)

        magnetization_simulations, rho_simulations=m_rho_iter_traj(num_simulations,H, op_dict,num_it,gamma,p,alpha)
        mean_fin[j],fin_m[j],stdev[j]=final_abs_magnetization(magnetization_simulations, rho_simulations)

    mean_magnet_multi,stdev_magnet_multi=moments(mean_fin)
    return mean_magnet_multi,stdev_magnet_multi


def m_vs_quant_multi_config(asymmetry,num_configuration,num_simulations,num_it, num_intervals,p,gamma,alpha,beta):
    """Plots magnetization versus density for a given initial network and for different values of gamma
    Graphs that can be plotted:
       Mean final magnetization versus gamma with its error bars (characteristic phase transition)
       Magnetization vs Rho (observable phase transition)
       Histograms of normalized magnetizations vs gamma (in order to verify bimodal presence)
       Variance of y coords of m vs binned x (possible secondary phase transiiton) """
    gamma_value=np.linspace(0.01,0.49,num_intervals)
    #beta_value=np.linspace(0.01,3.0,num_intervals)

    cmap = plt.cm.get_cmap('hsv', num_intervals+1)

    mean_magnet_multi=[0]*num_intervals
    stdev_magnet_multi=[0]*num_intervals
    
    fnoise_array=[0]*num_intervals

    pool = Pool(20)
    #a=pool.starmap(multi_config,zip(repeat(num_configuration),repeat(N),repeat(n),beta_value,repeat(asymmetry),repeat(num_simulations),repeat(num_it),repeat(gamma),repeat(p),repeat(alpha)))

    a=pool.starmap(multi_config,zip(repeat(num_configuration),repeat(N),repeat(n),repeat(beta),repeat(asymmetry),repeat(num_simulations),repeat(num_it),gamma_value,repeat(p),repeat(alpha)))
    mean_magnet_multi=[item[0] for item in a]
    stdev_magnet_multi=[item[1] for item in a]

            
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    #sg=str(gamma)
    #sg=[k for k in sg if k!='.']
    #sg="".join(sg)
    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)

    dirname = "../heterogeneous/off_multi_absm_vs_gamma_N_%d_beta_%s_n_%d_p_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d" %(N,sb,n,sp,sa,num_simulations,num_configuration,asymmetry)

    #dirname = "../heterogeneous/multi_absm_vs_beta_N_%d_gamma_%s_n_%d_p_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d" %(N,sg,n,sp,sa,num_simulations,num_configuration,asymmetry)

    if not os.path.exists(dirname):
        os.mkdir(dirname)
    #df = pandas.DataFrame(data={"beta_value": beta_value, "mean_magnet_multi": mean_magnet_multi, "stdev_magnet_multi":stdev_magnet_multi})
    df = pandas.DataFrame(data={"gamma_value": gamma_value, "mean_magnet_multi": mean_magnet_multi, "stdev_magnet_multi":stdev_magnet_multi})

    #df.to_csv(dirname+"/multi_absm_vs_beta_N_%d_gamma_%s_n_%d_p_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sg,n,sp,sa,num_simulations,num_configuration,asymmetry), sep=',',index=False)
    df.to_csv(dirname+"/multi_absm_vs_gamma_N_%d_beta_%s_n_%d_p_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sb,n,sp,sa,num_simulations,num_configuration,asymmetry), sep=',',index=False)


    a = 20 # fontsize
    # set up plot style
    plt_parameters = {
        "axes.labelsize": a,
        "axes.titlesize": a,
        "xtick.labelsize": a - 2,
        "ytick.labelsize": a - 2,
        "legend.framealpha": 0.8,
        "legend.fontsize": a - 4,
        "lines.linewidth": 2,
        # "axes.linewidth": 1.75,
    }
    plt.rcParams.update(plt_parameters)
    mean_magnet_multi=np.array(mean_magnet_multi)
    stdev_magnet_multi=np.array(stdev_magnet_multi)



    #plt.scatter(beta_value, mean_magnet_multi, s=10, color='crimson',label='Simulations')
    #plt.fill_between(beta_value, mean_magnet_multi -  stdev_magnet_multi, mean_magnet_multi +  stdev_magnet_multi,color='red', alpha=0.2)
    #plt.xlabel('β')

    plt.scatter(gamma_value, mean_magnet_multi, s=10, color='crimson',label='Simulations')
    plt.fill_between(gamma_value, mean_magnet_multi -  stdev_magnet_multi, mean_magnet_multi +  stdev_magnet_multi,color='red', alpha=0.2)
    plt.xlabel('γ')

    plt.ylabel("Absolute magnetization")
    plt.savefig(dirname+"/off_multi_absm_vs_gamma_N_%d_beta_%s_n_%d_p_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.png" %(N,sb,n,sp,sa,num_simulations,num_configuration,asymmetry), bbox_inches='tight')
    #plt.savefig(dirname+"/multi_absm_vs_beta_N_%d_gamma_%s_n_%d_p_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.png" %(N,sg,n,sp,sa,num_simulations,num_configuration,asymmetry), bbox_inches='tight')

    return

if __name__ == '__main__':
    
    #------------------------Parameters------------------------------#
    N=200 #number of nodes
    n=120 #number of hyperedges
    beta=1.5 #variance of exponential distribution
    
    gamma=0.49 #threshold for split
    p=0.5 #probability of split/merge for 2-edges
    alpha=1.0
    asymmetry=50
    num_it=10000 #number of iterations 
    num_simulations=50 #number of simulations for statistics
    num_intervals=50 #number of different gamma values for magnetization vs density function
    num_configuration=50
    
    #----------------------------------------------------------------#
    
    #H=heterogeneous_initial(N,n,beta)
    #op_dict=assign_opinions_asymmetry(N,asymmetry)
    #opinions_H=assign_hyper_opinions(op_dict,H)
    #size_H=[len(x) for x in H]
    m_vs_quant_multi_config(asymmetry,num_configuration,num_simulations,num_it, num_intervals,p,gamma,alpha,beta)    
    #abs_m_vs_gamma(num_simulations,H,N,asymmetry,beta,num_it, num_intervals,p,gamma,alpha)

