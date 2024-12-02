# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 09:18:52 2021

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
from scipy import stats

import time
from scipy.integrate import quad

import math
import os

import re

from plot_hmf import *

def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict


def alpha_gamma(alpha,gamma,edge):
    """Rescaling the given gamma depending on alpha.
    Used before the Decision function"""
    gamma=gamma*(len(edge))**(alpha-1)
    return gamma
    
def prob_size(n,beta):
    """Probability distribution for the size of the edges for the Heterogeneous Mean Field Case"""
    p=1/beta*np.exp(-(n-2)/beta)    
    return p
    
def hmf_magnetization(op_dict):
    """"Calculates the normalized magnetization of the hypergraph given the opinion dictionary"""
    m_node=0
    for i in range(len(op_dict)):
        m_node+=(2*op_dict[i]-1)
    return m_node/len(op_dict)

def hmf_influence(edge,op_edge,op_dict): 
    """"Implements the influence step for heterogeneous mean field.
    Updates the opinions array and the opinions edge on the minority vertices"""
    minor=minority(op_edge) #value of the minority opinion
    prob=op_edge.count(minor)/(len(op_edge)) #probability staying the same
    indices = [i for i, x in enumerate(op_edge) if x == minor] #which nodes are in the minority
    for i in range(len(indices)):
        if random.random()<=1-prob: #with probability 1-p, we change the opinion
            node=edge[indices[i]] #ID of the node with minority opinion
            op_dict[node]=op_dict[node]*(-1)+1
            op_edge=assign_hyperedge_op(op_dict,edge)
    return op_dict, op_edge

def hmf_two_influence(edge,op_edge,op_dict,p):
    """Adjusted Classical Adaptive Voter Model for HMF"""
    edge_fr=count_opinions(op_edge)
    if (edge_fr!=0 and edge_fr!=1): #if active
        if random.random()<=1-p: 
            node=random.randrange(0,2)
            op_dict[edge[node]]=op_dict[edge[node]]*(-1)+1
            op_edge=assign_hyperedge_op(op_dict,edge)
    return op_dict, op_edge


def hmf_dynamics(op_dict,beta,gamma_fixed,n_iter,p,alpha):
    """Implements the dynamics of HMF model"""
    N=len(op_dict)
    status='no' #Checks whether it is at equilibrium or not
    time_step=0 #a time step lasts until an edge of size n is picked 
    time_equilibrium=0
    n_array=np.zeros(N) #saves the frequency of the size of the picked edges
    
    
    while time_step<=n_iter and status!='equilibrium': #loops stop once reaches equilibrium or finishes iterations
        n=int(2+np.floor(np.random.exponential(scale=beta, size=None)))

        while n>len(op_dict): #if n> number of nodes then choose other n
            n=int(2+np.floor(np.random.exponential(scale=beta, size=None)))

        gamma=gamma_fixed        
        edge=random.sample(range(0, len(op_dict)), n)
        gamma=alpha_gamma(alpha,gamma,edge)
        op_edge=assign_hyperedge_op(op_dict,edge)
        edge_fr=count_opinions(op_edge)
        if len(op_edge)!=2: #if hyperedge: influence
            if gamma>0.5: #gamma must be <=0.5. For gamma>0.5, symmetry
                gamma_cor=1-gamma
            else:
                gamma_cor=gamma
            if edge_fr<= gamma_cor or edge_fr>=1-gamma_cor:
                n_array[n-2]+=1
                op_dict, op_edge=hmf_influence(edge,op_edge,op_dict)
            #else nothing because no structure
                
        elif len(op_edge)==2: #if simple edge: adaptation
            op_dict, op_edge=hmf_two_influence(edge,op_edge,op_dict,p)
            n_array[n-2]+=1
        time_step+=1
        if len(set([op_dict[i] for i in op_dict]))==1:
            print("Reached Equilibrium at iteration {:d}" .format(time_step))
            time_equilibrium=time_step
            status='equilibrium'
    time_equilibrium=time_step
    return op_dict, time_equilibrium, n_array

def hmf_dynamics_traj(op_dict,beta,gamma_fixed,n_iter,p,alpha):
    """Implements the HMF model SAVING ALL magnetizations (new output: m_time)"""
    N=len(op_dict)
    status='no' #Checks whether it is at equilibrium or not
    time_step=0 #a time step lasts until an edge of size n is picked 
    time_equilibrium=0
    n_array=np.zeros(N)
    m_time=[]
    while time_step<=n_iter and status!='equilibrium': #loops stop once reaches equilibrium or finishes iterations
        n=int(2+np.floor(np.random.exponential(scale=beta, size=None)))
        while n>len(op_dict): 
            n=int(2+np.floor(np.random.exponential(scale=beta, size=None)))
        gamma=gamma_fixed        
        edge=random.sample(range(0, len(op_dict)), n)
        gamma=alpha_gamma(alpha,gamma,edge)
        op_edge=assign_hyperedge_op(op_dict,edge)
        edge_fr=count_opinions(op_edge)
        if len(op_edge)!=2:
            if gamma>0.5:
                gamma_cor=1-gamma
            else:
                gamma_cor=gamma
            if edge_fr<= gamma_cor or edge_fr>=1-gamma_cor:
                n_array[n-2]+=1
                op_dict, op_edge=hmf_influence(edge,op_edge,op_dict)
                
        elif len(op_edge)==2:
            op_dict, op_edge=hmf_two_influence(edge,op_edge,op_dict,p)
            n_array[n-2]+=1
        time_step+=1
        m_time.append(hmf_magnetization(op_dict))
        if len(set([op_dict[i] for i in op_dict]))==1:
            print("Reached Equilibrium at iteration {:d}" .format(time_step))
            time_equilibrium=time_step
            status='equilibrium'
    time_equilibrium=time_step
    return op_dict, time_equilibrium, n_array, m_time

def hmf_multi_gamma(num_simulations, op_dict,beta,gamma,n_iter,p,alpha):
    """Calculates node-magnetization array with many simulations for a chosen gamma"""
    m_array=np.zeros(num_simulations)
    time_array=np.zeros(num_simulations)
    for i in range(num_simulations):
        print("%d th simulation" %(i))
        op_dict_init=copy.deepcopy(op_dict)
        op_dict_init, time_equilibrium, n_array= hmf_dynamics(op_dict_init,beta,gamma,n_iter,p,alpha)
        m_array[i]=(hmf_magnetization(op_dict_init))
        time_array[i]=np.log10(time_equilibrium)
    return m_array,time_array

def hmf_time_m(num_simulations, op_dict,beta,gamma,n_iter,p,alpha):
    """Calculates time evolution of node-magnetization array with for simulations for a chosen gamma"""

    for i in range(num_simulations):
        print("%d th simulation" %(i))
        op_dict_init=copy.deepcopy(op_dict)
        op_dict_init, time_equilibrium, n_array, m_time= hmf_dynamics_traj(op_dict_init,beta,gamma,n_iter,p,alpha)

    return m_time


def pool_average_magnet_sequence(num_simulations, num_config,beta,gamma,n_iter,p,N,alpha,asymmetry):
    "Calculates average magnetization for many configurations for many trajectories "
    m_average_list=np.zeros(num_config)
    m_stdev_list=np.zeros(num_config)
    t_average_list=np.zeros(num_config)
    t_stdev_list=np.zeros(num_config)

    for i in range(num_config):   
        op_dict= assign_opinions_asymmetry(N,asymmetry)
        m_array,time_array = hmf_multi_gamma(num_simulations, op_dict,beta,gamma,n_iter,p,alpha)
        m_average_list[i],m_stdev_list[i]=moments(m_array)
        t_average_list[i],t_stdev_list[i]=moments(time_array)

    t_aver,t_stdev=moments(t_average_list)
    m_aver,m_stdev=moments(m_average_list)

    return m_aver,m_stdev,t_aver,t_stdev
    
def pool_magnet_time_beta_varying(num_intervals,num_simulations, num_config,gamma,n_iter,p,alpha,asymmetry):
    "Calculates magnetization versus BETA for many configurations for many trajectories"
    beta=np.linspace(0.1,1.3,num_intervals)

    m_aver_multi=np.zeros(num_intervals)
    m_stdev_multi=np.zeros(num_intervals)
    t_aver_multi=np.zeros(num_intervals)
    t_stdev_multi=np.zeros(num_intervals)

    pool = Pool(20)

    a=pool.starmap(pool_average_magnet_sequence, zip(repeat(num_simulations), repeat(num_config),beta,repeat(gamma),repeat(n_iter),repeat(p),repeat(N),repeat(alpha),repeat(asymmetry)))
 
    m_aver_multi=[item[0] for item in a]
    m_stdev_multi=[item[1] for item in a]
    t_aver_multi=[item[2] for item in a]
    t_stdev_multi=[item[3] for item in a]
    
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)

    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)

    dirname = "../hmf/multi_config_varying_beta_N_%d_p_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d" %(N,sp,sg,sa,num_simulations,num_config,asymmetry)

    if not os.path.exists(dirname):
        os.mkdir(dirname)
        
    df = pandas.DataFrame(data={"beta": beta, "m_aver_multi": m_aver_multi, "m_stdev_multi":m_stdev_multi})
    df.to_csv(dirname+"/m_beta_N_%d_p_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sp,sg,sa,num_simulations,num_config,asymmetry), sep=',',index=False)    

    df = pandas.DataFrame(data={"beta": beta, "t_aver_multi": t_aver_multi, "t_stdev_multi":t_stdev_multi})
    df.to_csv(dirname+"/t_beta_N_%d_p_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sp,sg,sa,num_simulations,num_config,asymmetry), sep=',',index=False)    
   
    return 

def pool_magnet_time_gamma_varying(num_intervals,num_simulations, num_config,beta,n_iter,p,alpha,asymmetry):
    "Calculates magnetization versus GAMMA for many configurations for many trajectories"
    gamma=np.linspace(0.1,0.49,num_intervals)

    m_aver_multi=np.zeros(num_intervals)
    m_stdev_multi=np.zeros(num_intervals)
    t_aver_multi=np.zeros(num_intervals)
    t_stdev_multi=np.zeros(num_intervals)
    pool = Pool(20)

    a=pool.starmap(pool_average_magnet_sequence, zip(repeat(num_simulations), repeat(num_config),repeat(beta),gamma,repeat(n_iter),repeat(p),repeat(N),repeat(alpha),repeat(asymmetry)))
 
    m_aver_multi=[item[0] for item in a]
    m_stdev_multi=[item[1] for item in a]
    t_aver_multi=[item[2] for item in a]
    t_stdev_multi=[item[3] for item in a]
    
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)



    dirname = "../hmf/multi_config_varying_gamma_N_%d_p_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d" %(N,sp,sb,sa,num_simulations,num_config,asymmetry)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        
    df = pandas.DataFrame(data={"gamma": gamma, "m_aver_multi": m_aver_multi, "m_stdev_multi":m_stdev_multi})
    df.to_csv(dirname+"/m_gamma_N_%d_p_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sp,sb,sa,num_simulations,num_config,asymmetry), sep=',',index=False)    

    df = pandas.DataFrame(data={"gamma": gamma, "t_aver_multi": t_aver_multi, "t_stdev_multi":t_stdev_multi})
    df.to_csv(dirname+"/t_gamma_N_%d_p_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sp,sb,sa,num_simulations,num_config,asymmetry), sep=',',index=False)    
   
    return gamma, t_aver_multi, t_stdev_multi, m_aver_multi, m_stdev_multi


def pool_magnet_time_N_varying(num_intervals,num_simulations, num_config,beta,gamma,n_iter,p,alpha,asymmetry):
    "Magnetization versus N for many configurations for many trajectories"
    N=np.linspace(900,1000,num_intervals)
    N=[int(x) for x in N]
    pool = Pool(20)

    a=pool.starmap(pool_average_magnet_sequence, zip(repeat(num_simulations), repeat(num_config),repeat(beta),repeat(gamma),repeat(n_iter),repeat(p),N,repeat(alpha),repeat(asymmetry)))
 
    m_aver_multi=[item[0] for item in a]
    m_stdev_multi=[item[1] for item in a]
    t_aver_multi=[item[2] for item in a]
    t_stdev_multi=[item[3] for item in a]
    
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


    dirname = "../hmf/multi_config_varying_N_beta_%s_p_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d" %(sb,sp,sg,sa,num_simulations,num_config,asymmetry)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        
    df = pandas.DataFrame(data={"N": N, "m_aver_multi": m_aver_multi, "m_stdev_multi":m_stdev_multi})
    df.to_csv(dirname+"/m_N_beta_%s_p_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(sb,sp,sg,sa,num_simulations,num_config,asymmetry), sep=',',index=False)    

    df = pandas.DataFrame(data={"N": N, "t_aver_multi": t_aver_multi, "t_stdev_multi":t_stdev_multi})
    df.to_csv(dirname+"/t_N_beta_%s_p_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(sb,sp,sg,sa,num_simulations,num_config,asymmetry), sep=',',index=False)    
   
    return N, t_aver_multi, t_stdev_multi, m_aver_multi, m_stdev_multi

def pool_magnet_time_p_varying(num_intervals,num_simulations, num_config,beta,gamma,n_iter,alpha,asymmetry):
    "Magnetization versus a varying quantity for many configurations for many trajectories"

    p=np.linspace(0.05,1.0,num_intervals)

    m_aver_multi=np.zeros(num_intervals)
    m_stdev_multi=np.zeros(num_intervals)
    t_aver_multi=np.zeros(num_intervals)
    t_stdev_multi=np.zeros(num_intervals)
    pool = Pool(20)

    a=pool.starmap(pool_average_magnet_sequence, zip(repeat(num_simulations), repeat(num_config),repeat(beta),repeat(gamma),repeat(n_iter),p,repeat(N),repeat(alpha),repeat(asymmetry)))
 
    m_aver_multi=[item[0] for item in a]
    m_stdev_multi=[item[1] for item in a]
    t_aver_multi=[item[2] for item in a]
    t_stdev_multi=[item[3] for item in a]
    

    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    sa=str(alpha)
    sa=[k for k in sa if k!='.']
    sa="".join(sa)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)


    dirname = "../hmf/multi_config_varying_p_N_%d_beta_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d" %(N,sb,sg,sa,num_simulations,num_config,asymmetry)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        
    df = pandas.DataFrame(data={"p": p, "m_aver_multi": m_aver_multi, "m_stdev_multi":m_stdev_multi})
    df.to_csv(dirname+"/m_p_N_%d_beta_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sb,sg,sa,num_simulations,num_config,asymmetry), sep=',',index=False)    

    df = pandas.DataFrame(data={"beta": beta, "t_aver_multi": t_aver_multi, "t_stdev_multi":t_stdev_multi})
    df.to_csv(dirname+"/t_p_N_%d_beta_%s_gamma_%s_alpha_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sb,sg,sa,num_simulations,num_config,asymmetry), sep=',',index=False)    
   
    return p, t_aver_multi, t_stdev_multi, m_aver_multi, m_stdev_multi


def hmf_evolution_dynamics(op_dict,beta,gamma_fixed,n_iter,p,alpha):
    """Implements the HMF model. Memorizes the magnetization at each time step.  
    Same with before but used for hmf_evolution_multi_gamma"""
    N=len(op_dict)
    status='no' #Checks whether it is at equilibrium or not
    time_step=0 #a time step lasts until an edge of size n is picked 
    time_equilibrium=0
    n_array=np.zeros(N)
    
    
    m_time=[]

    
    while time_step<=n_iter and status!='equilibrium': #loops stop once reaches equilibrium or finishes iterations
        n=int(2+np.floor(np.random.exponential(scale=beta, size=None)))

        while n>len(op_dict):
            n=int(2+np.floor(np.random.exponential(scale=beta, size=None)))

        gamma=gamma_fixed        
        edge=random.sample(range(0, len(op_dict)), n)
        gamma=alpha_gamma(alpha,gamma,edge)
        op_edge=assign_hyperedge_op(op_dict,edge)
        edge_fr=count_opinions(op_edge)
        if len(op_edge)!=2:
            if gamma>0.5:
                gamma_cor=1-gamma
            else:
                gamma_cor=gamma
            if edge_fr<= gamma_cor or edge_fr>=1-gamma_cor:
                n_array[n-2]+=1
                op_dict, op_edge=hmf_influence(edge,op_edge,op_dict)
                
        elif len(op_edge)==2:
            op_dict, op_edge=hmf_two_influence(edge,op_edge,op_dict,p)
            n_array[n-2]+=1
        time_step+=1

        
        m_time.append(hmf_magnetization(op_dict))
        
            
        if len(set([op_dict[i] for i in op_dict]))==1:
            print("Reached Equilibrium at iteration {:d}" .format(time_step))
            time_equilibrium=time_step
            status='equilibrium'
    time_equilibrium=time_step
    return op_dict, time_equilibrium, n_array, m_time

def hmf_evolution_multi_gamma(num_simulations, op_dict,beta,gamma,n_iter,p,alpha):
    """Calculates node-magnetization array with different simulations 
    for a chosen gamma for every timestep"""
    m_time_array=[0]*num_simulations #each element is the time evolution of one trajectory
    time_array=np.zeros(num_simulations)
    for i in range(num_simulations):
        print("%d th simulation" %(i))
        op_dict_init=copy.deepcopy(op_dict)

        op_dict_init, time_equilibrium, n_array, m_time= hmf_evolution_dynamics(op_dict_init,beta,gamma,n_iter,p,alpha)
        # m_array[i]=abs(hmf_magnetization(op_dict_init))
        m_time_array[i]=m_time
        time_array[i]=np.log10(time_equilibrium)
    max_time=(len(max(m_time_array,key=len)))
    for item in m_time_array:
        if len(item) < max_time:
            item.extend([item[-1]] * (max_time - len(item))) #All trajectories that reach equilibrium freeze at equilibrium
    return m_time_array,time_array

    
def analytical_evolution_noiseless(N,beta,p,gamma,n_steps,asymmetry,noise):
    """Evolution of Analytical solution"""
    m=2*asymmetry/100-1
    m_sequence=np.zeros(n_steps)
    m2_sequence=np.zeros(n_steps)
    sum_hot=0
    for i in range(n_steps):
        print(i)
        sum_hot=0
        for n in range(3,N+1):
            mp=(1+m)/2
            mn=(1-m)/2
            vi=int(np.ceil(n*(1-gamma)))
            if vi<=n-1:
                for l in range(vi,(n-1)+1):
                    binomial=math.factorial(n-1)/(math.factorial(l)*math.factorial(n-1-l))
                    sum_hot+=prob_size(n,beta)/N*(binomial*l*((mp)**l*(mn)**(n-l)-(mn)**l*(mp)**(n-l)))
        m=m+2*sum_hot

        #WITHOUT NOISE
        m_sequence[i]=m
 
    return m_sequence

def analytical_evolution_noise(N,beta,p,gamma,n_steps,asymmetry,noise):
    """Evolution of Analytical solution"""
    m=2*asymmetry/100-1
    m_sequence=np.zeros(n_steps)
    m2_sequence=np.zeros(n_steps)
    sum_hot=0
    for i in range(n_steps):
        print(i)
        sum_hot=0
        for n in range(3,N+1):
            mp=(1+m)/2
            mn=(1-m)/2
            vi=int(np.ceil(n*(1-gamma)))
            if vi<=n-1:
                for l in range(vi,(n-1)+1):
                    binomial=math.factorial(n-1)/(math.factorial(l)*math.factorial(n-1-l))
                    sum_hot+=prob_size(n,beta)/N*(binomial*l*((mp)**l*(mn)**(n-l)-(mn)**l*(mp)**(n-l)))
        m=m+2*sum_hot
        #WITH NOISE
        m_sequence[i]=m*(1-2*noise)

 
    return m_sequence

def average_evol_magnet_sequence(num_simulations, num_config,beta,gamma,n_iter,p,N,alpha,asymmetry,noise):
    """Calculates the average magnetization of the trajectories for each time step with or without ESCAPE RATE"""
    m_average_list=[]
    m_stdv_list=[]
    

    final_magnet_array=np.zeros(num_config*num_simulations) #array to save final magnetizations. used for escape rate
    for i in range(num_config): 
        print('--------- Configuration %d ----------' %(i))
        op_dict=assign_opinions_asymmetry(N,asymmetry)
        m_time_array,time_array = hmf_evolution_multi_gamma(num_simulations, op_dict,beta,gamma,n_iter,p,alpha)

        for j in range(len(m_time_array)):
            t_x=list(range(len(m_time_array[j])))
            final_magnet_array[(i)*num_simulations+j]=m_time_array[j][-1]

        m_snap=list(zip_longest(*m_time_array)) #each element is an array of the magnetizations at a time step of many trajectories
        m_snap=[list(x) for x in m_snap]
        m_aver_snap=[0]*len(max(m_time_array,key=len)) #each element the average of the trajectories at a timestep
        m_stdv_snap=[0]*len(max(m_time_array,key=len))
        
        for i in range(len(max(m_time_array,key=len))):
            #Possible functions to calculate moments: moments, moments_none, moments_filter. Change it if needed:
            m_aver_snap[i],m_stdv_snap[i]=moments(m_snap[i])
            
        m_average_list.append(m_aver_snap) #each element is an array of the evolution of the average in that configuration
        m_stdv_list.append(m_stdv_snap)
    
    max_time=(len(max(m_average_list,key=len)))
    for item in m_average_list:
        if len(item) < max_time:
            item.extend([item[-1]] * (max_time - len(item)))
            
    m_average_list_snap=list(zip_longest(*m_average_list)) #each element is the averages of all configurations for a time step
    m_average_list_snap=[list(x) for x in m_average_list_snap]
    
    m_stdv_list_snap=list(zip_longest(*m_stdv_list)) #each element is an array of the magnetizations at a time step of many trajectories
    m_stdv_list_snap=[list(x) for x in m_stdv_list_snap]
    
    m_aver_final=[0]*len(m_average_list_snap)
    m_stdev_final=[0]*len(m_average_list_snap)
    for i in range(len(m_average_list_snap)):
        m_aver_final[i],m_stdev_final[i]=moments_none(m_average_list_snap[i])

    time_range=list(range(len(m_average_list_snap)))
    
 
                      
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
        
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/evol_comp_N_%d_p_%s_gamma_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d" %(N,sp,sg,sb,sa,num_simulations,num_config)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    df = pandas.DataFrame(data={"time_range": time_range, "m_aver_final": m_aver_final, "m_stdev_final":m_stdev_final})

    df.to_csv(dirname+"/evol_comp_N_%d_p_%s_gamma_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d.csv" %(N,sp,sg,sb,sa,num_simulations,num_config), sep=',',index=False)    

    return time_range, m_aver_final, m_stdev_final

def escape_rate_simulations(final_magnet_array,asymmetry):
    """Calculates fraction of trajectories that restore symmetry = pass the barrier: final magnetization has opposite sign"""
    num=0
    for i in range(len(final_magnet_array)):
        if asymmetry>50:
            if final_magnet_array[i]<= 0:
                num+=1
        elif asymmetry<50:
            if final_magnet_array[i]>=0:
                num+=1
    return num/len(final_magnet_array)

def escape_rate_simulations_w_var(final_magnet_array,asymmetry, num_sim, num_conf):
    """Calculates fraction of trajectories that restore symmetry = pass the barrier: final magnetization has opposite sign"""
    
    fr_array=np.zeros(num_conf)
    for i in range(num_config):
        num=0
        for j in range(num_sim):
            if asymmetry>50:
                if final_magnet_array[i*num_sim+j]<= 0:
                    num+=1
            elif asymmetry<50:
                if final_magnet_array[i*num_sim+j]>=0:
                    num+=1
        fr_array[i]=num/num_sim
    mean, stdev = moments(fr_array)
    return mean, stdev

def pool_r_calc_iteration_fit(num_simulations, num_config,beta,gamma,n_iter,p,N,alpha,asymmetry):
    """Calculates the escape rate experimentally"""
    m_average_list=[]
    m_stdv_list=[]

    final_magnet_array=np.zeros(num_config*num_simulations) #array to save final magnetizations. used for escape rate
    for i in range(num_config): 
        print('--------- Configuration %d ----------' %(i))
        op_dict=assign_opinions_asymmetry(N,asymmetry)
        m_time_array,time_array = hmf_evolution_multi_gamma(num_simulations, op_dict,beta,gamma,n_iter,p,alpha)

        for j in range(len(m_time_array)):
            t_x=list(range(len(m_time_array[j])))
            final_magnet_array[(i)*num_simulations+j]=m_time_array[j][-1]

    R=escape_rate_simulations(final_magnet_array,asymmetry)
    Rmean, Rstdv=escape_rate_simulations_w_var(final_magnet_array,asymmetry, num_simulations, num_config)
    # R_analytical=escape_rate_analytical(asymmetry,N,beta,gamma)
    
    m0=2*float(asymmetry)/100-1


    return R, barrier_kramer_N(N,beta,gamma,m0),curvature_kramer_N(m0,N,beta,gamma),curvature_kramer_N(0,N,beta,gamma), Rmean, Rstdv

def energy_integrand(m,N,beta,gamma,n,l):
    """Used to find an expression of the energy barrier. Integrand of RHS"""
    return ((1+m)/2)**(l)*((1-m)/2)**(n-l)-((1-m)/2)**l*((1+m)/2)**(n-l)

def barrier_kramer_N(N,beta,gamma,m0):
    """Rescaled no N: Calculates the energy barrier by integrating RHS. Limits: initial magnetization to 0. m0= number"""
    e_bar=0
    for n in range(3,N+1):
        v=(int(np.ceil(n*(1-gamma))))

        if v<n:
            for l in range(v,n):
                binomial=math.factorial(n-1)/(math.factorial(l)*math.factorial(n-1-l))
                # e_bar+=(-2)*prob_size(n,beta)/N*l*binomial*quad(energy_integrand, m0, 0, args=(N,beta,gamma,n,l))[0]
                e_bar+=(-2)*prob_size(n,beta)*l*binomial*quad(energy_integrand, m0, 0, args=(N,beta,gamma,n,l))[0]

    return e_bar

def exact_barrier_kramer_N(N,beta,gamma,m0,c):
    """Rescaled no N: Calculates the energy barrier by integrating RHS. Limits: initial magnetization to 0. m0= number"""
    e_bar=0
    for n in range(3,N+1):
        v=(int(np.ceil(n*(1-gamma))))

        if v<n:
            for l in range(v,n):
                binomial=math.factorial(n-1)/(math.factorial(l)*math.factorial(n-1-l))
                # e_bar+=(-2)*prob_size(n,beta)/N*l*binomial*quad(energy_integrand, m0, 0, args=(N,beta,gamma,n,l))[0]
                e_bar+=(-2)*prob_size(n,beta)*l*binomial*quad(energy_integrand, m0, c, args=(N,beta,gamma,n,l))[0]

    return e_bar

def exact_r_integrand(c,N,beta,gamma,m0,kT):
    return np.exp(exact_barrier_kramer_N(N,beta,gamma,m0,c)/kT)

def exact_r(N,beta,gamma,m0,kT,damping):
    integral1=quad(exact_r_integrand,-1, m0, args=(N,beta,gamma,m0,kT))[0]
    pbinom=(m0+1)/2
    delta=np.sqrt(pbinom*(1-pbinom)/N)
    mu=m0
    sigma=delta
    integral2=(2*np.pi*kT/(abs(curvature_kramer_N(m0,N,beta,gamma))))**(1/2)
    r = kT/damping*1/(integral1*integral2)
    return r

def curvature_kramer_N(m,N,beta,gamma):
    """Rescaled: No N: Calculates the curvature of energy function. Used for the Kramer's escape rate formula"""
    summ=0.0
    mp=(1+m)/2 #Positive
    mn=(1-m)/2 #Negative
    mt=(1+m)/(1-m) #Tan
    ma=(1-m)/(1+m) #Arctan
    

    for n in range(3,N+1):
        v=(int(np.ceil(n*(1-gamma))))
        if v<n:
            for l in range(v,n):
                binomial=math.factorial(n-1)/(math.factorial(l)*math.factorial(n-1-l))
                # summ+=2*prob_size(n,beta)/N*l*binomial*(l/2*mp**(l-1)*mn**(n-l-1)-(n-l-1)/2*mp**l*mn**(n-l-2)+l/2*(mn)**(l-1)*mp**(n-l-1)-(n-l-1)/2*mn**l*mp**(n-l-2))
                summ+=2*prob_size(n,beta)*l*binomial*(l/2*mp**(l-1)*mn**(n-l)-(n-l)/2*mp**l*mn**(n-l-1)+l/2*(mn)**(l-1)*mp**(n-l)-(n-l)/2*mn**l*mp**(n-l-1))
    return summ


def escape_rate_analytical_N(asymmetry,N,beta,gamma,kt,damping):
    """Calculates the Kramer's escape rate formula"""
    m0=2*(asymmetry)/100-1

    curv_a=curvature_kramer_N(m0,N,beta,gamma)
    curv_b=curvature_kramer_N(0,N,beta,gamma)
    Ebarrier=barrier_kramer_N(N,beta,gamma,m0)
    R=1/(2*np.pi*damping)*(abs(curv_a*curv_b))**(1/2)*np.exp(-Ebarrier/kt)
    return R

def escape_rate_analytical(asymmetry,N,beta,gamma,coef_array):

    """Calculates the Kramer's escape rate formula using the assumptions for proportionality of damping, kT"""
    m0=2*(asymmetry)/100-1

    curv_a=curvature_kramer_N(0,N,beta,gamma)
    curv_b=curvature_kramer_N(0,N,beta,gamma)
    Ebarrier=barrier_kramer_N(N,beta,gamma,m0)
    
    n=coef_array[0]
    m=coef_array[1]
    d=coef_array[2]
    h=coef_array[3]
    k=coef_array[4]
    l=coef_array[5]
    damping=beta**h*N**(k)*np.exp(l)
    kt=np.exp(d)*N**(n)*beta**(m)
    
    R=1/(2*np.pi*abs(damping))*(abs(curv_a*curv_b))**(1/2)*np.exp(-Ebarrier/kt)

    return R


def pool_fit_line_escape_rate(num_points, num_simulations, num_config,beta,gamma,n_iter,p,N,alpha,initial_as,final_as):
    """Linear fit to calculate kT and γ-damping coefficient of the Kramer's escape rate formula.
    ASSUMPTION: Curvature of m0 and 0 is the same
    Linearize system by taking log both sides.
    Output: Values of kT, γ-damping and fitted line wrt data"""
    
    R_array=np.zeros(num_points)
    lnR_array=np.zeros(num_points)
    Ebarrier_array=np.zeros(num_points)
    a_curvature_array=np.zeros(num_points)
    b_curvature_array=np.zeros(num_points)
    
    
    R_mean=np.zeros(num_points)
    R_stdv=np.zeros(num_points)

    asymmetry_array=np.linspace(initial_as,final_as,num_points)

    pool = Pool(20)

    a=pool.starmap(pool_r_calc_iteration_fit, zip(repeat(num_simulations), repeat(num_config),repeat(beta),repeat(gamma),repeat(n_iter),repeat(p),repeat(N),repeat(alpha),asymmetry_array))
 
    R_array=[item[0] for item in a]
    Ebarrier_array=[item[1] for item in a]
    a_curvature_array=[item[2] for item in a]
    b_curvature_array=[item[3] for item in a]
    R_mean=[item[4] for item in a]
    R_stdv=[item[5] for item in a]
    

    index=next((i for i, x in enumerate(R_array) if x==0), None)
    R_array=R_array[:index]
    R_mean=R_mean[:index]
    R_stdv=R_stdv[:index]
    asymmetry_array=asymmetry_array[:index]
    Ebarrier_array=Ebarrier_array[:index]
    lnR_array=[np.log(x) for x in R_array]

    pop, V = np.polyfit(Ebarrier_array, lnR_array, 1, cov=True) #fit straight line
    #residual=np.polyfit(Ebarrier_array, lnR_array, 1, full=True)[1][0] 
    res=scipy.stats.linregress(Ebarrier_array, lnR_array)
    R2=res.rvalue**2
    m=pop[0]
    b=pop[1]
    m_error=V[0][0]
    b_error=V[1][1]
    
    kt=-1/m 
    kt_error=abs(m_error/m**2)
    damping=curvature_kramer_N(0,N,beta,gamma)*np.exp(-b)/(2*np.pi)
    damping_error=abs(curvature_kramer_N(0,N,beta,gamma)*np.exp(-b)/(2*np.pi)*b_error)

    asymmetry_check=np.linspace(initial_as,final_as,100)
    
    m0_array=[2*x/100-1 for x in asymmetry_array]
    m0_check=[2*x/100-1 for x in asymmetry_check]
    R_check=[escape_rate_analytical_N(x,N,beta,gamma,kt,damping) for x in asymmetry_check]

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
        

    dirname = "../hmf/Kramer_N_%d_p_%s_gamma_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d" %(N,sp,sg,sb,sa,num_simulations,num_config)
    
    
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    


    #df = pandas.DataFrame(data={"asymmetry_array": asymmetry_array, "R_array":R_array, "kt":kt, "damping":damping,"kt_error":kt_error, "damping_error":damping_error})
    df = pandas.DataFrame(data={"m0_array": m0_array, "R_array":R_array,  "R_mean":R_mean, "R_stdv":R_stdv, "kt":kt, "damping":damping,"kt_error":kt_error, "damping_error":damping_error,"R2":R2})

    df.to_csv(dirname+"/Kramer_N_%d_p_%s_gamma_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d.csv" %(N,sp,sg,sb,sa,num_simulations,num_config), sep=',',index=False)

    df1 = pandas.DataFrame(data={"m0_check": m0_check, "R_check": R_check})
    df1.to_csv(dirname+"/Kramer_analytical_N_%d_p_%s_gamma_%s_beta_%s_alpha_%s_nsim_%d_nconf_%d.csv" %(N,sp,sg,sb,sa,num_simulations,num_config), sep=',',index=False)

    
    return m0_array, R_mean, R_stdv, m0_check, R_check
    
def pool_discord_m_vs_alpha(num_intervals,num_simulations, num_config,beta,gamma,n_iter,p,asymmetry):
    "Calculates magnetization versus alpha discordance parameter for many configurations for many trajectories"
    alpha=np.linspace(0.0,1.0,num_intervals)
    m_aver_multi=np.zeros(num_intervals)
    m_stdev_multi=np.zeros(num_intervals)
    t_aver_multi=np.zeros(num_intervals)
    t_stdev_multi=np.zeros(num_intervals)

    pool = Pool(20)

    a=pool.starmap(pool_average_magnet_sequence, zip(repeat(num_simulations), repeat(num_config),repeat(beta),repeat(gamma),repeat(n_iter),repeat(p),repeat(N),alpha,repeat(asymmetry)))
 
    m_aver_multi=[item[0] for item in a]
    m_stdev_multi=[item[1] for item in a]
    t_aver_multi=[item[2] for item in a]
    t_stdev_multi=[item[3] for item in a]
    
    
    sp=str(p)
    sp=[k for k in sp if k!='.']
    sp="".join(sp)
    sb=str(beta)
    sb=[k for k in sb if k!='.']
    sb="".join(sb)
    sg=str(gamma)
    sg=[k for k in sg if k!='.']
    sg="".join(sg)
        
    dirname = "../hmf/discord_alpha_vs_m_N_%d_beta_%s_p_%s_gamma_%s_nsim_%d_nconf_%d_asym_%d" %(N,sb,sp,sg,num_simulations,num_config,asymmetry)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
   
    df = pandas.DataFrame(data={"alpha": alpha, "m_aver_multi": m_aver_multi, "m_stdev_multi":m_stdev_multi})
    df.to_csv(dirname+"/disc_m_vs_alpha_N_%d_beta_%s_p_%s_gamma_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sb,sp,sg,num_simulations,num_config,asymmetry), sep=',',index=False)    

    dg = pandas.DataFrame(data={"alpha": alpha, "t_aver_multi": t_aver_multi, "t_stdev_multi":t_stdev_multi})
    dg.to_csv(dirname+"/disc_t_vs_alpha_N_%d_beta_%s_p_%s_gamma_%s_nsim_%d_nconf_%d_asym_%d.csv" %(N,sb,sp,sg,num_simulations,num_config,asymmetry), sep=',',index=False)    


    return  



def potential_well_for_plot(N,beta,gamma,x):
    """Calculates the energy barrier by integrating RHS. Limits: initial magnetization to 0. m0= number"""
    e_bar=0
    for n in range(3,N+1):
        v=(int(np.ceil(n*(1-gamma))))

        if v<n:
            for l in range(v,n):
                binomial=math.factorial(n-1)/(math.factorial(l)*math.factorial(n-1-l))
                # e_bar+=(-2)*prob_size(n,beta)/N*l*binomial*quad(energy_integrand, m0, 0, args=(N,beta,gamma,n,l))[0]
                e_bar+=(-2)*prob_size(n,beta)*l*binomial*quad(energy_integrand, 0, x, args=(N,beta,gamma,n,l))[0]
    return e_bar


def supersheet_scaling():
    """Extracts kT and λ from fitting Kramer s escape rate for many simulations of escape rate vs m0
    Output: csv file with kT and λ for many N and many β"""
    dirname="../Plots/hmf/Analytical/closer to transition/Corrected/supercomputer/data_scaling/"
    
    beta_list=[]
    N_list=[]
    kt_list=[]
    kt_error_list=[]
    damping_list=[]
    damping_error_list=[]
    
    for filename in os.listdir(dirname):
        print(filename)
        filename=dirname+filename+"/"+filename+".csv"
        mat = re.match(r'../Plots/hmf/Analytical/closer to transition/Corrected/supercomputer/data_scaling/Kramer_N_(.*)_beta_(.*)_gamma_(.*)_p_(.*)/Kramer_N_(.*)_beta_(.*)_gamma_(.*)_p_(.*).csv', filename)
        N=int(mat.group(1))
        if mat.group(2).startswith('0') and len(mat.group(2))>2:
            beta=float(mat.group(2))/100
        else:
            beta=float(mat.group(2))/10        

        
        gamma=float(mat.group(3))/10
        p=float(mat.group(4))/10
        data=pandas.read_csv(filename)
        asymmetry_array=data['asymmetry_array'].values
        
        R_array=data['R_array'].values
        asymmetry_array=asymmetry_array[np.logical_not(np.isnan(asymmetry_array))]
        R_array=R_array[np.logical_not(np.isnan(R_array))]
        index=next((i for i, x in enumerate(R_array) if x==0), None)
        R_array=R_array[:index]
        asymmetry_array=asymmetry_array[:index]
        
        m0_array=[2*x/100-1 for x in asymmetry_array]
        Ebarrier_array=[barrier_kramer_N(N,beta,gamma,x) for x in m0_array]
        
        lnR_array=[np.log(x) for x in R_array]
        pop, V = np.polyfit(Ebarrier_array, lnR_array, 1, cov=True) #fit straight line
        m=pop[0]
        b=pop[1]
        m_error=V[0][0]
        b_error=V[1][1]
        
        kt=-1/m 
        kt_error=abs(m_error/m**2)
        damping=curvature_kramer_N(0,N,beta,gamma)*np.exp(-b)/(2*np.pi)
        damping_error=abs(curvature_kramer_N(0,N,beta,gamma)*np.exp(-b)/(2*np.pi)*b_error)
        
        N_list.append(N)
        beta_list.append(beta)
        kt_list.append(kt)
        kt_error_list.append(kt_error)
        damping_list.append(damping)
        damping_error_list.append(damping_error)
    
    
    lnkt_list=[np.log(x) for x in kt_list]
    lnbeta_list=[np.log(x) for x in beta_list]
    lndamping_list=[np.log(x) for x in damping_list]
    lnkt_error_list=[kt_error_list[i]/kt_list[i] for i in range(len(kt_list))]
    lndamping_error_list=[damping_error_list[i]/damping_list[i] for i in range(len(damping_list))]
    dirname = "../Plots/hmf/Analytical/closer to transition/Corrected/supersheet_scaling" 
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        
    df = pandas.DataFrame(data={ "N_list": N_list,"beta_list": beta_list,"kt_list":kt_list, "kt_error_list":kt_error_list, "damping_list":damping_list, "damping_error_list":damping_error_list,    "lnbeta_list":lnbeta_list, "lnkt_list":lnkt_list, "lnkt_error_list":lnkt_error_list,"lndamping_list":lndamping_list, "lndamping_error_list":lndamping_error_list })
    df.to_csv(dirname+"/supersheet_scaling.csv", sep=',',index=False)    
    
    return N_list, lnbeta_list, lnkt_list, lnkt_error_list, lndamping_list, lndamping_error_list

if __name__ == '__main__':
    
    #-------------------------Parameters-------------------------------
    N=100 #number of nodes
    p= 0.5 #0.5 #probability of split/merge for 2-edges
    n_iter= 4000000 #number of iterations
    num_simulations= 100
    num_intervals= 100 #number of points in varying parameters 
    num_config=50
    beta= 3.0 #mean 2+beta (we have displaced the prob distr), stdv= beta
    gamma= 0.3
    alpha= 0.5 #Different dependence of discordance on size of edges !! DEFAULT: alpha=1 !!
    asymmetry=float(55)
    num_points=30 #USED FOR ESCAPE RATE FIT, number of points in escape rate versus quantity
    
    #------------------------------------------------------------------
    #pool_magnet_time_beta_varying(num_intervals,num_simulations, num_config,gamma,n_iter,p,alpha,asymmetry)
    pool_magnet_time_gamma_varying(num_intervals,num_simulations, num_config,beta,n_iter,p,alpha,asymmetry)
    #pool_magnet_time_N_varying(num_intervals,num_simulations, num_config,beta,gamma,n_iter,p,alpha,asymmetry)
    #pool_magnet_time_p_varying(num_intervals,num_simulations, num_config,beta,gamma,n_iter,alpha,asymmetry)
    #pool_fit_line_escape_rate(num_points, num_simulations, num_config,beta,gamma,n_iter,p,N,alpha,55,70)
    #pool_discord_m_vs_alpha(num_intervals,num_simulations, num_config,beta,gamma,n_iter,p,asymmetry)
