# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 13:32:16 2021

@author: Nikos
"""


import numpy as np
import random
import copy
from tools import minority, count_opinions,  assign_hyperedge_op,  assign_opinions_asymmetry, moments, moments_none, moments_filter
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.optimize import fsolve
import pandas
import pandas
from multiprocessing import Pool
import multiprocessing
from itertools import repeat
from itertools import zip_longest
import scipy

import time
from scipy.integrate import quad
from scipy import stats
import math
import os

# from hmf import potential_well_for_plot

def plot_hmf_time_m(m_time):
    """Plots time evolution of  node-magnetization array with for simulations for a chosen gamma"""    
    plt.plot(m_time)
    a = 18  # fontsize
    # set up plot style
    plt_parameters = {
        "axes.labelsize": a,
        "axes.titlesize": a,
        "xtick.labelsize": a - 2,
        "ytick.labelsize": a - 2,
        "legend.framealpha": 0.8,
        "legend.fontsize": a - 5,
        "lines.linewidth": 2,
        "axes.linewidth": 1.75,
    }
    plt.rcParams.update(plt_parameters)
    plt.xlabel('Timesteps')
    plt.ylabel('Magnetization')
    plt.show()
    return


def plot_one_config_magnet_time_vs_gamma(gamma, t_mean, t_stdev, m_mean, m_stdev, num_intervals, num_simulations, op_dict,beta,n_iter,p,alpha,N):
    """Plots scatter+error bar |m|, log10t vs gamma given an initial configuration"""
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


    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)

    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/one_config_varying_gamma_N_%d_p_%s_beta_%s_alpha_%s_nsim_%d" %(N,sp,sb,sa,num_simulations)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    plt.scatter(gamma, m_mean, s=10, color='crimson',label='Mean')
    plt.fill_between(gamma, m_mean -  m_stdev, m_mean +  m_stdev,color='red', alpha=0.2)
    plt.xlabel('$\gamma$')
    plt.ylabel("Magnetization")   

    plt.savefig(dirname+"/magnetization_vs_gamma_N_%d_p_%s_beta_%s_alpha_%s_nsim_%d.png" %(N,sp,sb,sa,num_simulations), bbox_inches='tight')
    plt.show()

    plt.scatter(gamma, t_mean, s=10, color='crimson',label='Mean')
    plt.fill_between(gamma, t_mean -  t_stdev, t_mean +  t_stdev,color='red', alpha=0.2)
    plt.ylabel('$\log_{10}(t_{equil})$')       
    plt.xlabel('$\gamma$')


    plt.savefig(dirname+"/log10t_vs_gamma_N_%d_p_%s_beta_%s_alpha_%s_nsim_%d.png" %(N,sp,sb,sa,num_simulations), bbox_inches='tight')
    plt.show()

    return


def one_config_plot_magnet_time_vs_beta(beta, t_mean, t_stdev, m_mean, m_stdev, num_intervals, num_simulations, op_dict,gamma,n_iter,p,alpha,N):
    """Plots scatter+error bar |m|, log10t vs gamma given an initial configuration"""
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


    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)
    
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/one_config_varying_beta_N_%d_p_%s_gamma_%s_alpha_%s_nsim_%d" %(N,sp,sg,sa,num_simulations)
    if not os.path.exists(dirname):
        os.mkdir(dirname)


    plt.scatter(beta, m_mean, s=10, color='crimson',label='Mean')
    plt.fill_between(beta, m_mean -  m_stdev, m_mean +  m_stdev,color='red', alpha=0.2)
    plt.xlabel('$\\beta$')
    plt.ylabel("Magnetization")   


    plt.savefig(dirname+"/magnetization_vs_beta_N_%d_p_%s_gamma_%s_alpha_%s_nsim_%d.png" %(N,sp,sg,sa,num_simulations), bbox_inches='tight')
    plt.show()

    plt.scatter(beta, t_mean, s=10, color='crimson',label='Mean')
    plt.fill_between(beta, t_mean -  t_stdev, t_mean +  t_stdev,color='red', alpha=0.2)
    plt.ylabel('$\log_{10}(t_{equil})$')       
    plt.xlabel('$\\beta$')


    plt.savefig(dirname+"/log10t_vs_beta_N_%d_p_%s_gamma_%s_alpha_%s_nsim_%d.png" %(N,sp,sg,sa,num_simulations), bbox_inches='tight')
    plt.show()
 

    return


def plot_magnet_time_beta_varying(beta, t_aver_multi, t_stdev_multi, m_aver_multi, m_stdev_multi,num_intervals,num_simulations, num_config,gamma,n_iter,p,alpha,asymmetry,N):
    "Plots magnetization versus BETA for many configurations for many trajectories"
    
    a = 20 # fontsize
    # set up plot style
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)

    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)
    
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/multi_config_varying_beta_N_%d_p_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d" %(N,sp,sg,sa,num_simulations,num_config,asymmetry)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
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
    plt.scatter(beta, m_aver_multi, s=10, color='crimson',label='Mean')
    plt.fill_between(beta, m_aver_multi -  m_stdev_multi, m_aver_multi +  m_stdev_multi,color='red', alpha=0.2)
    plt.xlabel('$\\beta$')
    plt.ylabel("Magnetization") 
    plt.savefig(dirname+"/m_vs_beta_N_%d_p_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.png" %(N,sp,sg,sa,num_simulations,num_config,asymmetry), bbox_inches='tight')

    plt.show()
 
    plt.scatter(beta, t_aver_multi, s=10, color='crimson',label='Mean')
    plt.fill_between(beta, t_aver_multi -  t_stdev_multi, t_aver_multi +  t_stdev_multi,color='red', alpha=0.2)
    plt.xlabel('$\\beta$')
    plt.ylabel("$\log_{10}(t_{equil})$")  
    plt.savefig(dirname+"/log10t_vs_beta_N_%d_p_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.png" %(N,sp,sg,sa,num_simulations,num_config,asymmetry), bbox_inches='tight')

    plt.show()    

    return


def plot_magnet_time_gamma_varying(gamma, t_aver_multi, t_stdev_multi, m_aver_multi, m_stdev_multi,num_intervals,num_simulations, num_config,beta,n_iter,p,alpha,asymmetry,N):
    a = 20 # fontsize
    # set up plot style
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)

    
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/multi_config_varying_gamma_N_%d_p_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d" %(N,sp,sb,sa,num_simulations,num_config,asymmetry)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
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
    plt.scatter(gamma, m_aver_multi, s=10, color='crimson',label='Mean')
    plt.fill_between(gamma, m_aver_multi -  m_stdev_multi, m_aver_multi +  m_stdev_multi,color='red', alpha=0.2)
    plt.xlabel('$\gamma$')
    plt.ylabel("Magnetization") 
    plt.savefig(dirname+"/m_vs_gamma_N_%d_p_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.png" %(N,sp,sb,sa,num_simulations,num_config,asymmetry), bbox_inches='tight')

    plt.show()
 
    plt.scatter(gamma, t_aver_multi, s=10, color='crimson',label='Mean')
    plt.fill_between(gamma, t_aver_multi -  t_stdev_multi, t_aver_multi +  t_stdev_multi,color='red', alpha=0.2)
    plt.xlabel('$\gamma$')
    plt.ylabel("$\log_{10}(t_{equil})$")  
    plt.savefig(dirname+"/log10t_vs_gamma_N_%d_p_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.png" %(N,sp,sb,sa,num_simulations,num_config,asymmetry), bbox_inches='tight')

    plt.show()    

    return

def plot_magnet_time_N_varying(N, t_aver_multi, t_stdev_multi, m_aver_multi, m_stdev_multi,num_intervals,num_simulations, num_config,beta,gamma,n_iter,p,alpha,asymmetry):
    a = 20 # fontsize
    # set up plot style
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)
    
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/multi_config_varying_N_beta_%s_p_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d" %(sb,sp,sg,sa,num_simulations,num_config,asymmetry)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
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
    plt.scatter(N, m_aver_multi, s=10, color='crimson',label='Mean')
    plt.fill_between(N, m_aver_multi -  m_stdev_multi, m_aver_multi +  m_stdev_multi,color='red', alpha=0.2)
    plt.xlabel('N')
    plt.ylabel("Magnetization") 
    plt.savefig(dirname+"/m_vs_N_beta_%s_p_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.png" %(sb,sp,sg,sa,num_simulations,num_config,asymmetry), bbox_inches='tight')

    plt.show()
 
    plt.scatter(N, t_aver_multi, s=10, color='crimson',label='Mean')
    plt.fill_between(N, t_aver_multi -  t_stdev_multi, t_aver_multi +  t_stdev_multi,color='red', alpha=0.2)
    plt.xlabel('N')
    plt.ylabel("$\log_{10}(t_{equil})$")  
    plt.savefig(dirname+"/log10t_vs_N_beta_%s_p_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.png" %(sb,sp,sg,sa,num_simulations,num_config,asymmetry), bbox_inches='tight')

    plt.show()    

    return


def plot_magnet_time_p_varying(p, t_aver_multi, t_stdev_multi, m_aver_multi, m_stdev_multi,num_intervals,num_simulations, num_config,gamma,n_iter,beta,alpha,asymmetry,N):
    a = 20 # fontsize
    # set up plot style

    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)
    
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/multi_config_varying_p_N_%d_beta_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d" %(N,sb,sg,sa,num_simulations,num_config,asymmetry)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
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
    plt.scatter(p, m_aver_multi, s=10, color='crimson',label='Mean')
    plt.fill_between(p, m_aver_multi -  m_stdev_multi, m_aver_multi +  m_stdev_multi,color='red', alpha=0.2)
    plt.xlabel('p')
    plt.ylabel("Magnetization") 
    plt.savefig(dirname+"/m_vs_p_N_%d_beta_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.png" %(N,sb,sg,sa,num_simulations,num_config,asymmetry), bbox_inches='tight')

    plt.show()
 
    plt.scatter(p, t_aver_multi, s=10, color='crimson',label='Mean')
    plt.fill_between(p, t_aver_multi -  t_stdev_multi, t_aver_multi +  t_stdev_multi,color='red', alpha=0.2)
    plt.xlabel('p')
    plt.ylabel("$\log_{10}(t_{equil})$")  
    plt.savefig(dirname+"/log10t_vs_p_N_%d_beta_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.png" %(N,sb,sg,sa,num_simulations,num_config,asymmetry), bbox_inches='tight')

    plt.show()    

    return


def plot_fit_line_escape_rate(m0_array, R_mean, R_stdv, m0_check, R_check, num_points, num_simulations, num_config,beta,gamma,n_iter,p,N,alpha,initial_as,final_as):
    """Plots escape rate versus initial asymmetry for both analytical and simulations"""
    
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)
        
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/Kramer_N_%d_p_%s_gamma_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d" %(N,sp,sg,sb,sa,num_simulations,num_config)
    
    
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

    plt.scatter(m0_array, R_mean, s=10, color='crimson',label='Simulations')
    plt.fill_between(m0_array, R_mean -  R_stdv, R_mean +  R_stdv,color='red', alpha=0.2)
   
    plt.plot(m0_check,R_check,label='Analytical', c="black", zorder=2)
    plt.legend()
    plt.xlabel('Initial Magnetization')
    plt.ylabel("Escape Rate")
    plt.savefig(dirname+"/Kramer_N_%d_p_%s_gamma_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d.png" %(N,sp,sg,sb,sa,num_simulations,num_config), bbox_inches='tight')

    plt.show()

    return


def plot_saturation_prediction_N(m_predict,coef_array, N_simulations, m_simulations, m_stdev, beta,gamma, p, alpha, asymmetry):
    """Plots analytical and simulations of final magnetization vs N"""
    
    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/Prediction_N_p_%s_gamma_%s_beta_%s_alpha_%s_asymmetry_%d" %(sp,sg,sb,sa,asymmetry)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
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
    plt.scatter(N_simulations,m_predict,c='black',label='Analytical')
    plt.scatter(N_simulations,m_simulations, c='crimson', label='Simulations')
    plt.fill_between(N_simulations, m_simulations -  m_stdev, m_simulations +  m_stdev,color='red', alpha=0.2)

    # plt.scatter(N_simulations, m_simulations, c='black', label='Simulations')
    plt.xlabel('N')
    plt.ylabel('Magnetization')
    plt.legend(loc="lower right")

        
    plt.savefig(dirname+"/Constant_beta_%s_gamma_%s_p_%s.png" %(sb,sg,sp), bbox_inches='tight')


    return m_predict



def plot_prediction_evolution(m_predict,coef_array, N, m_simulations,m_stdev, beta_simulations,gamma, p, alpha, asymmetry):
    """Plots analytical and simulations of final magnetization vs beta"""

    
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/Prediction_beta_N_%d_p_%s_gamma_%s_alpha_%s_asymmetry_%d" %(N,sp,sg,sa,asymmetry)
    if not os.path.exists(dirname):
        os.mkdir(dirname)


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
    # plt.scatter(beta_simulations,m_predict,c='orange',label='Prediction')
    # plt.scatter(beta_simulations, m_simulations, c='black', label='Simulations')
    plt.scatter(beta_simulations,m_predict,c='black',label='Analytical')
    plt.scatter(beta_simulations,m_simulations, c='crimson', label='Simulations')
    plt.fill_between(beta_simulations, m_simulations -  m_stdev, m_simulations +  m_stdev,color='red', alpha=0.2)

    plt.xlabel('β')
    plt.ylabel('Magnetization')
    plt.legend(loc='lower right')

        
    plt.savefig(dirname+"/Constant_beta_N_%d_gamma_%s_p_%s.png" %(N,sg,sp), bbox_inches='tight')

    plt.show()
    return 



def plot_comparison_prediction(m_sequence,m_aver_final,m_stdev_final, N,beta,p,gamma,asymmetry,alpha):
    "Compares analytical and computational TRAJECTORIES"


    n_steps=len(m_aver_final)
    time_range=np.arange(n_steps)
    
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)
        
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/Prediction_evolution_beta_%s_N_%d_p_%s_gamma_%s_alpha_%s_asymmetry_%d" %(sb,N,sp,sg,sa,asymmetry)
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
    time_range=np.array(time_range)
    m_aver_final=np.array(m_aver_final)
    m_stdev_final=np.array(m_stdev_final)
    m_sequence=np.array(m_sequence)
    plt.scatter(time_range, m_aver_final, s=10, color='crimson',label='Simulations')
    plt.fill_between(time_range, m_aver_final -  m_stdev_final, m_aver_final +  m_stdev_final,color='red', alpha=0.2)
   

    plt.scatter(time_range,m_sequence,s=4,label='Analytical', c="black", zorder=2)
    plt.legend(loc="lower right")
    plt.xlabel('Time')
    plt.ylabel("Magnetization")   
    
    plt.savefig(dirname+"/Prediction_beta_N_%d_p_%s_gamma_%s_alpha_%s_asymmetry_%d.png" %(N,sp,sg,sa,asymmetry), bbox_inches='tight')

    plt.show()
    return

def plot_potential_well(N,beta,gamma):
    """Plots potential well by numerical integration"""
    x=np.linspace(-1,1,100)
    # print(potential_well_for_plot(N,beta,gamma,-1))
    # print(potential_well_for_plot(N,beta,gamma,1))
    # print(energy_integrand(-1,N,beta,gamma,2,1))
    # print(energy_integrand(-1,N,beta,gamma,2,1))
    y=[potential_well_for_plot(N,beta,gamma,x_pt) for x_pt in x]
    # y=[energy_integrand(x_pt,N,beta,gamma,9,5) for x_pt in x]
    plt.plot(x,y)
    plt.xlabel('x')
    plt.ylabel('U(x)')    
    plt.show()

def plot_size_distribution():
    """Plots size distribution of the edges"""
    x=np.arange(2,10,0.01)
    pos=5
    lw=4
    beta=3.0
    n_array=[1/beta*np.exp(-(n-2)/beta) for n in x]
    plt.plot(x,n_array,label='β=3',linewidth=lw,c='red')
    # plt.scatter(x,n_array,label='β=3',c='red')
    beta=0.5
    n_array=[1/beta*np.exp(-(n-2)/beta) for n in x]
    plt.plot(x,n_array,label='β=0.5',linewidth=lw,c='blue')
    # plt.scatter(x,n_array,label='β=0.5',c='blue')

    plt.vlines(x=pos, ymin=min(n_array), ymax=max(n_array), color='green', linestyle='--', linewidth=lw-2, label='x=3',zorder=3)
    n=pos
    plt.hlines(y=1/beta*np.exp(-(n-2)/beta), xmin=2.0,xmax=pos,color='gold', linestyle='--', linewidth=lw-2, label='y=%.3f' %(1/beta*np.exp(-(n-2)/beta)),zorder=3)
    n=pos
    beta=3
    plt.hlines(y=1/beta*np.exp(-(n-2)/beta), xmin=2.0,xmax=pos,color='indigo', linestyle='--', linewidth=lw-2, label='y=%.3f' %(1/beta*np.exp(-(n-2)/beta)),zorder=3)    
    

    a = 25  # fontsize
    # set up plot style
    plt_parameters = {
        "axes.labelsize": a,
        "axes.titlesize": a,
        "xtick.labelsize": a - 2,
        "ytick.labelsize": a - 2,
        "legend.framealpha": 0.8,
        "legend.fontsize": a - 5,
        "lines.linewidth": 2,
        "axes.linewidth": 1.75,
    }
    plt.rcParams.update(plt_parameters)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel('x')
    plt.ylabel('π(x)')    
    return


def plot_discord_m_vs_alpha(alpha,m_aver_multi, m_stdv_multi, t_aver_multi, t_stdv_multi,num_intervals,num_simulations, num_config,beta,gamma,n_iter,p,asymmetry,N):
    "Plots magnetization versus alpha discordance parameter for many configurations for many trajectories"

    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)
        
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/discord_alpha_vs_m_N_%d_beta_%s_p_%s_gamma_%s_nsim_%d_nconf_%d_asym_%d" %(N,sb,sp,sg,num_simulations,num_config,asymmetry)
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
    plt.scatter(alpha, m_aver_multi, s=10, color='crimson',label='Mean')
    plt.fill_between(alpha, m_aver_multi -  m_stdv_multi, m_aver_multi +  m_stdv_multi,color='red', alpha=0.2)
    plt.xlabel('$\\alpha$')
    plt.ylabel("Magnetization") 
    plt.savefig(dirname+"/discord_m_vs_alpha_N_%d_beta_%s_p_%s_gamma_%s_nsim_%d_nconf_%d_asym_%d.png" %(N,sb,sp,sg,num_simulations,num_config,asymmetry), bbox_inches='tight')

    plt.show()
    
    plt.scatter(alpha, t_aver_multi, s=10, color='crimson',label='Mean')
    plt.fill_between(alpha, t_aver_multi -  t_stdv_multi, t_aver_multi +  t_stdv_multi,color='red', alpha=0.2)
    plt.xlabel('$\\alpha$')
    plt.ylabel('$\log_{10}(t_{equil})$')       
    plt.savefig(dirname+"/discord_t_vs_alpha_N_%d_beta_%s_p_%s_gamma_%s_nsim_%d_nconf_%d_asym_%d.png" %(N,sb,sp,sg,num_simulations,num_config,asymmetry), bbox_inches='tight')

    plt.show()
    return



def plot_finite_size_analysis():
    """Plots magnetization vs beta for many different sizes N"""

        
    data=pandas.read_csv("data_analysis_ N_vs_m_for_beta=3_asymmetry=0_68.csv")
    beta_100=data['beta_100'].values	
    m_aver__100	=data['m_aver__100'].values
    beta_200=data['beta_200'].values
    m_aver__200=data['m_aver__200'].values				
    beta_300=data['beta_300'].values	
    m_aver__300=data['m_aver__300'].values				
    beta_400=data['beta_400'].values	
    m_aver__400=data['m_aver__400'].values
    beta_500=data['beta_500'].values	
    m_aver__500=data['m_aver__500'].values				
    beta_600=data['beta_600'].values	
    m_aver__600=data['m_aver__600'].values				
    beta_700=data['beta_700'].values	
    m_aver__700=data['m_aver__700'].values				
    beta_800=data['beta_800'].values	
    m_aver__800=data['m_aver__800'].values			
    beta_900=data['beta_900'].values	
    m_aver__900=data['m_aver__900'].values				
    beta_1000=data['beta_1000'].values	
    m_aver__1000=data['m_aver__1000'].values				
    beta_1100=data['beta_1100'].values	
    m_aver__1100=data['m_aver__1100'].values				
    beta_1200=data['beta_1200'].values	
    m_aver__1200=data['m_aver__1200'].values				
    beta_1300=data['beta_1300'].values	
    m_aver__1300=data['m_aver__1300'].values				
    beta_1400=data['beta_1400'].values	
    m_aver__1400=data['m_aver__1400'].values				
    beta_1500=data['beta_1500'].values	
    m_aver__1500=data['m_aver__1500'].values	
    
    
    a = 15  # fontsize
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
    
    # NUM_COLORS = 15
    
    
    cm = plt.cm.hsv(np.linspace(0,1,30))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color',plt.cm.hsv(np.linspace(0,1,15)))
    

    plt.scatter(beta_100, m_aver__100, s=30)
    plt.scatter(beta_200, m_aver__200, s=30)
    plt.scatter(beta_300, m_aver__300, s=30)
    plt.scatter(beta_400, m_aver__400, s=30)
    plt.scatter(beta_500, m_aver__500, s=30)
    plt.scatter(beta_600, m_aver__600, s=30)
    plt.scatter(beta_700, m_aver__700, s=30)
    plt.scatter(beta_800, m_aver__800, s=30)
    plt.scatter(beta_900, m_aver__900, s=30)
    plt.scatter(beta_1000, m_aver__1000, s=30)
    plt.scatter(beta_1100, m_aver__1100, s=30)
    plt.scatter(beta_1200, m_aver__1200, s=30)
    plt.scatter(beta_1300, m_aver__1300, s=30)
    plt.scatter(beta_1400, m_aver__1400, s=30)
    plt.scatter(beta_1500, m_aver__1500, s=30)
    
    plt.plot(beta_100, m_aver__100, label="N=100")
    plt.plot(beta_200, m_aver__200, label="N=200")
    plt.plot(beta_300, m_aver__300, label="N=300")
    plt.plot(beta_400, m_aver__400, label="N=400")
    plt.plot(beta_500, m_aver__500, label="N=500")
    plt.plot(beta_600, m_aver__600, label="N=600")
    plt.plot(beta_700, m_aver__700, label="N=700")
    plt.plot(beta_800, m_aver__800, label="N=800")
    plt.plot(beta_900, m_aver__900, label="N=900")
    plt.plot(beta_1000, m_aver__1000, label="N=1000")
    plt.plot(beta_1100, m_aver__1100, label="N=1100")
    plt.plot(beta_1200, m_aver__1200, label="N=1200")
    plt.plot(beta_1300, m_aver__1300, label="N=1300")
    plt.plot(beta_1400, m_aver__1400, label="N=1400")
    plt.plot(beta_1500, m_aver__1500, label="N=1500")
    
    
    
    plt.xlabel("β ")
    plt.ylabel("Magnetization") 
    # plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.legend(ncol=2)
    
    plt.show()
    
    #FSA for m0=0.1
    data=pandas.read_csv("data_analysis_asymmetry_0_55.csv")
    
    x100=data['x100'].values	
    y100=data['y100'].values
    x200=data['x200'].values	
    y200=data['y200'].values	
    x300=data['x300'].values	
    y300=data['y300'].values	
    x400=data['x400'].values	
    y400=data['y400'].values	
    x500=data['x500'].values	
    y500=data['y500'].values	
    x900=data['x900'].values	
    y900=data['y900'].values	
    x1000=data['x1000'].values	
    y1000=data['y1000'].values	
    x1200=data['x1200'].values	
    y1200=data['y1200'].values	
    x1500=data['x1500'].values	
    y1500=data['y1500'].values	

    x100, y100 = zip(*sorted(zip(x100, y100)))
    x200, y200 = zip(*sorted(zip(x200, y200)))
    x300, y300 = zip(*sorted(zip(x300, y300)))
    x400, y400 = zip(*sorted(zip(x400, y400)))
    x500, y500 = zip(*sorted(zip(x500, y500)))
    x900, y900 = zip(*sorted(zip(x900, y900)))
    x1000, y1000 = zip(*sorted(zip(x1000, y1000 )))
    x1200, y1200 = zip(*sorted(zip(x1200, y1200)))
    x1500, y1500 = zip(*sorted(zip(x1500, y1500)))
    

    cm = plt.cm.hsv(np.linspace(0,1,30))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_prop_cycle('color',plt.cm.hsv(np.linspace(0,1,9)))
    
    a = 15  # fontsize
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
        
    plt.scatter(x100, y100, s=30)
    plt.plot(x100, y100, label="N=100")
    
    plt.scatter(x200, y200, s=30)
    plt.plot(x200, y200, label="N=200")
    
    plt.scatter(x300, y300, s=30)
    plt.plot(x300, y300, label="N=300")
    
    plt.scatter(x400, y400, s=30)
    plt.plot(x400, y400, label="N=400")
    
    plt.scatter(x500, y500, s=30)
    plt.plot(x500, y500, label="N=500")
    
    plt.scatter(x900, y900, s=30)
    plt.plot(x900, y900, label="N=900")
    
    plt.scatter(x1000, y1000, s=30)
    plt.plot(x1000, y1000, label="N=1000")

    plt.scatter(x1200, y1200, s=30)
    plt.plot(x1200, y1200, label="N=1200")
    
    plt.scatter(x1500, y1500, s=30)
    plt.plot(x1500, y1500, label="N=1500")
    
    
    plt.xlabel("β ")
    plt.ylabel("Magnetization") 
    plt.legend(ncol=2)
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


def plot_supersheet_scaling(N_list, lnbeta_list, lnkt_list, lnkt_error_list, lndamping_list, lndamping_error_list): 
    """Plots lnkT vs lnβ and lnλ vs lnβ and uses linear regression for slope and intercept
       Used to extract coefficients of scaling relations VOL 1 """
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/supersheet_scaling" 
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    N_unique=list(set(N_list))
    N_unique_significant=[x for x in N_unique if x >=100]
    cm = plt.cm.hsv(np.linspace(0,1,len(N_unique)))

    kt_slope=[]
    kt_slope_error=[]
    damp_slope=[]
    damp_slope_error=[]
    kt_inter=[]
    kt_inter_error=[]
    damp_inter=[]
    damp_inter_error=[]
    N_list_analysis=[]
    R2_kt=[]
    R2_damping=[]
    for i in range(len(N_unique)):
        lnkt_size=[]
        lnkt_error_size=[]
        lnb_size=[]
        lndamping_size=[]
        lndamping_error_size=[]
        for j in range(len(N_list)):
            if N_list[j]==N_unique[i] and lnbeta_list[j]>=0 and lnbeta_list[j]<=1.0 and N_list[j]>=100:
             
                lnkt_size.append(lnkt_list[j])
                lnb_size.append(lnbeta_list[j])
                lnkt_error_size.append(lnkt_error_list[j])
        if len(lnkt_size)>0:   
            a = 22  # fontsize
            # set up plot style
            plt_parameters = {
                "axes.labelsize": a,
                "axes.titlesize": a,
                "xtick.labelsize": a - 2,
                "ytick.labelsize": a - 2,
                "legend.framealpha": 0.8,
                "legend.fontsize": a - 10,
                "lines.linewidth": 2,
                "axes.linewidth": 1.75,
            }
            plt.rcParams.update(plt_parameters)
                 
            plt.errorbar(lnb_size, lnkt_size,lnkt_error_size, ls="None", linestyle='None', capsize=5, capthick=1,alpha=0.6,zorder=1,color=cm[i])
            plt.scatter(lnb_size, lnkt_size, s=30, label="N=%d" %(N_unique[i]),alpha=1,zorder=5,color=cm[i])
            plt.plot(lnb_size, np.poly1d(np.polyfit(lnb_size, lnkt_size, 1))(lnb_size),alpha=0.4,zorder=2,color=cm[i])
            coeff, cov=np.polyfit(lnb_size, lnkt_size, 1, cov=True)
            res=scipy.stats.linregress(lnb_size, lnkt_size)
            R2_kt.append(res.rvalue**2)
            kt_slope.append(coeff[0])
            kt_inter.append(coeff[1])
            kt_slope_error.append(cov[0][0])
            kt_inter_error.append(cov[1][1])
            
  
    plt.xlabel("ln(β)")
    plt.ylabel("ln(kT)") 

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig(dirname+"/lnkt_vs_lnbeta.png" , bbox_inches='tight')
      
    plt.show()
    
    for i in range(len(N_unique)):

        lnb_size=[]
        lndamping_size=[]
        lndamping_error_size=[]
        for j in range(len(N_list)):
            if N_list[j]==N_unique[i] and lnbeta_list[j]>=0 and lnbeta_list[j]<=1.0 and N_list[j]>=100:
                
                lnb_size.append(lnbeta_list[j])
                lndamping_size.append(lndamping_list[j])
                lndamping_error_size.append(lndamping_error_list[j])

        if len(lndamping_size)>0:   
            a = 22  # fontsize
            # set up plot style
            plt_parameters = {
                "axes.labelsize": a,
                "axes.titlesize": a,
                "xtick.labelsize": a - 2,
                "ytick.labelsize": a - 2,
                "legend.framealpha": 0.8,
                "legend.fontsize": a - 10,
                "lines.linewidth": 2,
                "axes.linewidth": 1.75,
            }
            plt.rcParams.update(plt_parameters)
                    
            plt.errorbar(lnb_size, lndamping_size,lndamping_error_size, ls="None", linestyle='None', capsize=5, capthick=1,alpha=0.6,zorder=1,color=cm[i])
            plt.scatter(lnb_size, lndamping_size, s=30, label="N=%d" %(N_unique[i]),alpha=1,zorder=5,color=cm[i])
            N_list_analysis.append(N_unique[i])
            plt.plot(lnb_size, np.poly1d(np.polyfit(lnb_size, lndamping_size, 1))(lnb_size),alpha=0.4,zorder=2,color=cm[i])
            coeff, cov=np.polyfit(lnb_size, lndamping_size, 1, cov=True)
            res=scipy.stats.linregress(lnb_size, lndamping_size)
            R2_damping.append(res.rvalue**2)
            damp_slope.append(coeff[0])
            damp_inter.append(coeff[1])
            damp_slope_error.append(cov[0][0])
            damp_inter_error.append(cov[1][1])

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel("ln(β)")
    plt.ylabel("ln(λ)")          
    plt.savefig(dirname+"/lndamping_vs_lnbeta.png" , bbox_inches='tight')

    plt.show()
    
    df = pandas.DataFrame(data={     "N_list_analysis":N_list_analysis,"kt_slope": kt_slope, "kt_slope_error":kt_slope_error, "kt_inter":kt_inter, "kt_inter_error":kt_inter_error, "damp_slope":damp_slope, "damp_slope_error":damp_slope_error,  "damp_inter":damp_inter, "damp_inter_error":damp_inter_error, "R2_damping":R2_damping, "R2_kt":R2_kt })
    df.to_csv(dirname+"/supersheet_inter_slopes.csv", sep=',',index=False)    
        
    return

    
def scaling_coefficients(N_list_analysis, kt_slope, kt_slope_error, kt_inter,kt_inter_error,damp_slope,damp_slope_error,damp_inter,damp_inter_error):
    """Plots kT intercept vs lnN and λ intercept vs lnβ and uses linear regression for slope and intercept
       Used to extract coefficients of scaling relations VOL 2
       Output: plots AND CSV FILE WITH coefficients of scaling relations"""
       
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/supersheet_scaling" 
    if not os.path.exists(dirname):
        os.mkdir(dirname)


    print(N_list_analysis)
    logN=[np.log(x) for x in N_list_analysis]
    coeff, cov=np.polyfit(logN, kt_inter, 1, cov=True)


    f = lambda x, a, b: a*x + b  # function to fit
    # fit with initial guess for parameters [1, 1]
    
    xb=np.linspace(min(logN),max(logN),100)
    pars1, corr1 = scipy.optimize.curve_fit(f, logN, kt_inter, [1, 1], kt_inter_error)    
    a_kt, b_kt = pars1
    a_error_kt,b_error_kt=corr1[0,0], corr1[1,1]
    res=scipy.stats.linregress(logN, kt_inter)
    R2_kt=res.rvalue**2    

            
    plt.plot(xb, a_kt*xb + b_kt, 'r')

    # plt.errorbar(logN, kt_inter, kt_inter_error, ls="None", linestyle='None', capsize=5, capthick=1,alpha=0.6,zorder=1)
    plt.scatter(logN, kt_inter, s=15,alpha=1,zorder=5)
    plt.xlabel("ln(N)")
    plt.ylabel("kT intercept")  
    plt.savefig(dirname+"/kt_interc_vs_lnN.png" , bbox_inches='tight')
  
    plt.show()


    xb=np.linspace(min(logN),max(logN),100)
    pars2, corr2 = scipy.optimize.curve_fit(f, logN, damp_inter, [1, 1], damp_inter_error)    
    res=scipy.stats.linregress(logN, damp_inter)
    R2_damp=res.rvalue**2   
    
    a_damp, b_damp = pars2
    a_error_damp,b_error_damp=corr2[0,0], corr2[1,1]

    plt.plot(xb, a_damp*xb + b_damp, 'r')
    plt.errorbar(logN, damp_inter,damp_inter_error, ls="None", linestyle='None', capsize=5, capthick=1,alpha=0.6,zorder=1)
    plt.scatter(logN, damp_inter, s=15,alpha=1,zorder=5)
    plt.xlabel("ln(N)")  
    plt.ylabel("$\lambda$ intercept")  
    plt.savefig(dirname+"/lambda_interc_vs_lnN.png" , bbox_inches='tight')
  
    plt.show()
    
    aver_ktslope, stdev_ktslope=moments(kt_slope)
    aver_damslope, stdev_damslope=moments(damp_slope)

    coefficients=[a_kt,aver_ktslope,b_kt,aver_damslope,a_damp,b_damp]
    coefficients_error=[a_error_kt,b_error_kt,a_error_damp,b_error_damp,stdev_ktslope,stdev_damslope]

    df = pandas.DataFrame(data={     "coefficients":coefficients,"coefficients_error": coefficients_error, "R2_kt":R2_kt, "R2_damp":R2_damp })
    df.to_csv(dirname+"/coefficients.csv", sep=',',index=False)    
            

    return coefficients, coefficients_error

def plot_beta_scaling_extended(N_list, lnbeta_list, lnkt_list, lnkt_error_list, lndamping_list, lndamping_error_list):
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/beta_scaling_extended" 
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    N_unique=list(set(N_list))
    N_unique_significant=[x for x in N_unique if x >=100]
    cm = plt.cm.hsv(np.linspace(0,1,len(N_unique)))

    kt_slope=[]
    kt_slope_error=[]
    damp_slope=[]
    damp_slope_error=[]
    kt_inter=[]
    kt_inter_error=[]
    damp_inter=[]
    damp_inter_error=[]
    N_list_analysis=[]
    for i in range(len(N_unique)):
        lnkt_size=[]
        lnkt_error_size=[]
        lnb_size=[]
        lndamping_size=[]
        lndamping_error_size=[]
        for j in range(len(N_list)):
            if N_list[j]==N_unique[i]  and (N_list[j]==50 or N_list[j]>=100) and N_list[j]!=110 and N_list[j]!=120 and N_list[j]!=180:
             
                lnkt_size.append(lnkt_list[j])
                lnb_size.append(lnbeta_list[j])
                lnkt_error_size.append(lnkt_error_list[j])
        if len(lnkt_size)>0:   
            a = 22  # fontsize
            # set up plot style
            plt_parameters = {
                "axes.labelsize": a,
                "axes.titlesize": a,
                "xtick.labelsize": a - 2,
                "ytick.labelsize": a - 2,
                "legend.framealpha": 0.8,
                "legend.fontsize": a - 10,
                "lines.linewidth": 2,
                "axes.linewidth": 1.75,
            }
            plt.rcParams.update(plt_parameters)
                    
            # plt.errorbar(lnb_size, lnkt_size,lnkt_error_size, ls="None", linestyle='None', capsize=5, capthick=1,alpha=0.6,zorder=1,color=cm[i])
            plt.scatter(lnb_size, lnkt_size, s=15, label="N=%d" %(N_unique[i]),alpha=1,zorder=5,color=cm[i])

  
    plt.xlabel("ln(β)")
    plt.ylabel("ln(kT)") 

    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.savefig(dirname+"/lnkt_vs_lnbeta.png" , bbox_inches='tight')
      
    plt.show()
    
    for i in range(len(N_unique)):

        lnb_size=[]
        lndamping_size=[]
        lndamping_error_size=[]
        for j in range(len(N_list)):
            if N_list[j]==N_unique[i] and (N_list[j]==50 or N_list[j]>=100) and N_list[j]!=110 and N_list[j]!=120 and N_list[j]!=180:
                
                lnb_size.append(lnbeta_list[j])
                lndamping_size.append(lndamping_list[j])
                lndamping_error_size.append(lndamping_error_list[j])

        if len(lndamping_size)>0:   
            a = 22  # fontsize
            # set up plot style
            plt_parameters = {
                "axes.labelsize": a,
                "axes.titlesize": a,
                "xtick.labelsize": a - 2,
                "ytick.labelsize": a - 2,
                "legend.framealpha": 0.8,
                "legend.fontsize": a - 10,
                "lines.linewidth": 2,
                "axes.linewidth": 1.75,
            }
            plt.rcParams.update(plt_parameters)
                    
            # plt.errorbar(lnb_size, lndamping_size,lndamping_error_size, ls="None", linestyle='None', capsize=5, capthick=1,alpha=0.6,zorder=1,color=cm[i])
            plt.scatter(lnb_size, lndamping_size, s=15, label="N=%d" %(N_unique[i]),alpha=1,zorder=5,color=cm[i])
            N_list_analysis.append(N_unique[i])


    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.xlabel("ln(β)")
    plt.ylabel("ln(λ)")          
    plt.savefig(dirname+"/lndamping_vs_lnbeta.png" , bbox_inches='tight')

    plt.show()
    
    return 