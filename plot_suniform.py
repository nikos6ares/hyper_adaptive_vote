# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:15:04 2021

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

import pandas
import os

def plot_abs_m_vs_gamma(mean_fin_array,stdev_array,gamma_value,num_simulations,num_it, num_intervals,p,alpha,N,S,n,asymmetry):
    """Plots average and stdev of absolute final magnetization over many simulations 
    for the same initial configuration for varying gamma""" 
    
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    dirname = "../Plots/s_uniform/abs_m_vs_gamma_p_%s_N_%d_S_%d_n_%d_asym_%d_alpha_%s_nsim_%d" %(sp,N,S,n,asymmetry,sa,num_simulations)
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
    
    gamma_value=np.array(gamma_value)
    mean_fin_array=np.array(mean_fin_array)
    stdev_array=np.array(stdev_array)
    
    plt.scatter(gamma_value, mean_fin_array, s=10, color='crimson',label='Simulations')
    plt.fill_between(gamma_value, mean_fin_array -  stdev_array, mean_fin_array +  stdev_array,color='red', alpha=0.2)

    plt.xlabel('γ')

    plt.ylabel("Absolute magnetization")   
    plt.savefig(dirname+"/abs_m_vs_gamma_p_%s_N_%d_S_%d_n_%d_asym_%d_alpha_%s_nsim_%d.png" %(sp,N,S,n,asymmetry,sa,num_simulations), bbox_inches='tight')

    plt.show()
    return 
    
    
def plot_symmetry_cap(N_array, noise_cap, noise_cap_stdev,mean_degree,p,S,n,asymmetry,gamma,alpha,num_simulations):
    
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
   
    a = 18  # fontsize
    # set up plot style
    plt_parameters = {
        "axes.labelsize": a,
        "axes.titlesize": a,
        "xtick.labelsize": a - 2,
        "ytick.labelsize": a - 2,
        "legend.framealpha": 0.8,
        "legend.fontsize": a - 4,
        "lines.linewidth": 2,
        "axes.linewidth": 1.75,
    }
    plt.rcParams.update(plt_parameters)
        
    plt.plot(N_array, noise_cap, marker='o', color='blue',zorder=2,label="mean")
    
    
    plt.plot(N_array, noise_cap_stdev, marker='o', color='red',zorder=2, label="stdev")
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("Symmetry Restoration") 
    plt.savefig(dirname+"/symmetry_restoration_meandeg_%d_p_%s_S_%d_n_%d_asym_%d_gamma_%s_alpha_%s_nsim_%d.png" %(mean_degree,sp,S,n,asymmetry,sg,sa,num_simulations), bbox_inches='tight')

    plt.show()
    return


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

def magnetization_vs_density(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it, num_intervals,alpha,N,S,n,asymmetry,p):
    """Plots edge-based magnetization versus density for a given initial network and 
    for different values of gamma """
    # gamma=np.linspace(0.01,0.499,num_intervals)
    gamma=0.2
    p=np.linspace(0.7,0.9,num_intervals)
    rho_simulations=[0]*num_intervals
    magnetization_simulations=[0]*num_intervals
    cmap = plt.cm.get_cmap('hsv', num_intervals+1)
    
    for i in range(num_intervals):
        # magnetization_simulations[i], rho_simulations[i]=m_rho_iter_traj(num_simulations,hypergraph, op_dict,num_it,gamma[i],p,alpha)
        magnetization_simulations[i], rho_simulations[i]=m_rho_iter_traj(num_simulations,hypergraph, op_dict,num_it,gamma,p[i],alpha)
        print(p[i])
        for j in range(num_simulations):
                rho_active=[1-x for x in rho_simulations[i][j]]
                # plt.scatter(magnetization_simulations[i][j],rho_active, s=10,alpha=1.0,color=cmap(i),label="γ= %.2f" %gamma[i] if j == 0 else "",zorder=num_intervals-i)
                # plt.plot(magnetization_simulations[i][j],rho_active,'-o',alpha=0.4,color=cmap(i),label="γ= %.2f" %gamma[i] if j == 0 else "",zorder=num_intervals-i)
                plt.scatter(magnetization_simulations[i][j],rho_active, s=10,alpha=0.5,color=cmap(i),label="p= %.2f" %p[i] if j == 0 else "",zorder=num_intervals+i)
    


    # sp=str(p)
    # sp=[k for k in sp if k!='.']
    # sp="".join(sp)
    # sa=str(alpha)
    # sa=[k for k in sa if k!='.']
    # sa="".join(sa)

    # dirname = "../Plots/s_uniform/m_vs_rho_p_%s_N_%d_S_%d_n_%d_asym_%d_alpha_%s_nsim_%d_ninter_%d" %(sp,N,S,n,asymmetry,sa,num_simulations,num_intervals)
    # if not os.path.exists(dirname):
        # os.mkdir(dirname)


    a = 20  # fontsize
    # set up plot style
    plt_parameters = {
        "axes.labelsize": a,
        "axes.titlesize": a,
        "xtick.labelsize": a - 2,
        "ytick.labelsize": a - 2,
        "legend.framealpha": 0.8,
        "legend.fontsize": a - 4,
        "lines.linewidth": 2,
        "axes.linewidth": 1.75,
    }
    plt.rcParams.update(plt_parameters)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.ylabel("Density of active edges")
    plt.xlabel("Magnetization")
    # plt.savefig(dirname+"/m_vs_rho_p_%s_N_%d_S_%d_n_%d_asym_%d_alpha_%s_nsim_%d_ninter_%d.png" %(sp,N,S,n,asymmetry,sa,num_simulations,num_intervals), bbox_inches='tight')

    plt.show()
    return

def plot_bands_on():
    """Puts blue lines of predicted values from theorem for MERGE/REWIRE switched ON or OFF """

    data=pandas.read_csv("ON_absm_vs_gamma_N_100_S_10_n_20_p_05_alpha_10_nsim_301.csv")
    # data=pandas.read_csv("OFF_absm_vs_gamma_N_100_S_10_n_20_p_05_alpha_10_nsim_301.csv")

    gamma_value=data['gamma_value'].values	
    mean_fin_array	=data['mean_fin_array'].values
    stdev_array = data['stdev_array'].values
    plt.fill_between(gamma_value, mean_fin_array -  stdev_array, mean_fin_array +  stdev_array,color='red', alpha=0.2)
    plt.scatter(gamma_value,mean_fin_array, label='Simulations',color="crimson", zorder=3, s=5)
    
    xcoords = [2/10, 3/10, 4/10, 5/10]
    for xc in xcoords:
        print(xc)
        plt.axvline(x=xc, color='blue', linestyle='--', linewidth=1.0, label='Analytical' if xc == xcoords[0] else '')
    a = 20 # fontsize
    # set up plot style
    plt_parameters = {
        "axes.labelsize": a,
        "axes.titlesize": a,
        "xtick.labelsize": a - 2,
        "ytick.labelsize": a - 2,
        "legend.framealpha": 0.8,
        "legend.fontsize": a - 7,
        "lines.linewidth": 2,
        # "axes.linewidth": 1.75,
    }
    plt.rcParams.update(plt_parameters)
    plt.legend()
    plt.xlabel("γ")
    plt.ylabel("Absolute Magnetization") 
    plt.show()
    return