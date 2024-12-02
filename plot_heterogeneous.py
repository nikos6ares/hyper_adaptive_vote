# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:46:17 2021

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


def plot_abs_m_vs_gamma_one_config(gamma_value, mean_fin_array, stdev_array,num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it, num_intervals,p,gamma,alpha,beta,N,n):
    """Plots absolute magnetization versus gamma"""

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
    mean_fin_array=np.array(mean_fin_array)
    stdev_array=np.array(stdev_array)
    
    
    
    plt.scatter(gamma_value, mean_fin_array, s=10, color='crimson',label='Simulations')
    plt.fill_between(gamma_value, mean_fin_array -  stdev_array, mean_fin_array +  stdev_array,color='red', alpha=0.2)

    plt.xlabel('γ')

    plt.ylabel("Absolute magnetization")   
    
    plt.savefig(dirname+"/absm_vs_gamma_N_%d_n_%d_p_%s_alpha_%s_nsim_%d_ninterv_%d.png" %(N,n,sp,sa,num_simulations,num_intervals), bbox_inches='tight')
    plt.show()
    return 


def plot_m_vs_gamma_multi_config(gamma_value, mean_magnet_multi, stdev_magnet_multi,asymmetry,num_configuration,num_simulations,num_it, num_intervals,p,alpha,beta,N,n):
    """Calculates magnetization versus gamma for many trajectories for many configurations"""


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
    
    
    
    # plt.scatter(beta_value, mean_magnet_multi, s=10, color='crimson',label='Simulations')
    # plt.fill_between(beta_value, mean_magnet_multi -  stdev_magnet_multi, mean_magnet_multi +  stdev_magnet_multi,color='red', alpha=0.2)
    # plt.xlabel('β')

    plt.scatter(gamma_value, mean_magnet_multi, s=10, color='crimson',label='Simulations')
    plt.fill_between(gamma_value, mean_magnet_multi -  stdev_magnet_multi, mean_magnet_multi +  stdev_magnet_multi,color='red', alpha=0.2)
    plt.xlabel('γ')

    plt.ylabel("Absolute magnetization")   
    plt.savefig(dirname+"/absm_vs_gamma_N_%d_n_%d_p_%s_beta_%s_alpha_%s_nsim_%d_ninterv_%d.png" %(N,n,sp,sb,sa,num_simulations,num_intervals), bbox_inches='tight')
    plt.show()
    return 


def plot_m_vs_beta_multi_config(beta_value, mean_magnet_multi, stdev_magnet_multi,asymmetry,num_configuration,num_simulations,num_it, num_intervals,p,alpha,gamma,N,n):
    """Calculates magnetization versus beta for many trajectories for many configurations"""


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
    
    
    
    plt.scatter(beta_value, mean_magnet_multi, s=10, color='crimson',label='Simulations')
    plt.fill_between(beta_value, mean_magnet_multi -  stdev_magnet_multi, mean_magnet_multi +  stdev_magnet_multi,color='red', alpha=0.2)
    plt.xlabel('β')


    plt.ylabel("Absolute magnetization")   
    plt.savefig(dirname+"/absm_vs_beta_N_%d_n_%d_p_%s_gamma_%s_alpha_%s_nsim_%d_ninterv_%d.png" %(N,n,sp,sg,sa,num_simulations,num_intervals), bbox_inches='tight')
    plt.show()
    return 