    # -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:04:52 2021

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
import numpy.ma as ma
from itertools import zip_longest
    

from tools import assign_hyper_opinions, magnetization_density, count_opinions, init_impact_edges,node_magnet_density
from dynamics import decide, alpha_gamma

def iterations(hypergraph,op_dict,num_it,gamma,p,alpha):
    """Iterates decide for num_it times with input as output at every step
    The m_array and rho_array have the magnetization of the whole hypergraph for each iteration"""
    opinions_hypergraph=assign_hyper_opinions(op_dict,hypergraph)
    m_array=[]
    rho_array=[]
    
    #------------------------- Uncomment NODE or EDGE based magnetization (default edge)--------------
    kind_magnet="edge"
    # kind_magnet="node"
    #-----------------------------------------------------------------------------
    
    
    for i in range(num_it):
        id_edge=randrange(len(hypergraph)) #chooses a random hyperedge of the hypergraph
        
        if kind_magnet=="edge":
            m, rho = magnetization_density(opinions_hypergraph)
        elif kind_magnet=="node":
            m, rho = node_magnet_density(opinions_hypergraph,op_dict)
        
        m_array.append(m)
        rho_array.append(rho)
        hypergraph, opinions_hypergraph,op_dict=decide(id_edge,hypergraph,opinions_hypergraph,op_dict,gamma,p,alpha) #split or influence
        if all([len(set(num)) == 1 for num in opinions_hypergraph]):
            print("Reached Equilibrium at iteration {:d}" .format(i))
            
            if kind_magnet=="edge":
                m, rho = magnetization_density(opinions_hypergraph)
            elif kind_magnet=="node":
                m, rho = node_magnet_density(opinions_hypergraph,op_dict)
        
            m_array.append(m)
            rho_array.append(rho)
            break
        if i==num_it-1:
            print("Did not reach equilibrium")
    return hypergraph,opinions_hypergraph,op_dict, m_array, rho_array
        

def statistics_calc(hypergraph, opinions_hypergraph, op_dict,num_it,gamma,p,alpha):
    """"Calculates for the final iterated hypergraph the following:
        num_species = number of edges
        size_speces = array with the size of each edge
        ratio_ones = the fraction of nodes with ones in the whole hypergraph
        m_array = same- unchanged m_array with iterations function
        rho_array = same - unchanged rho_array with iterations function
        num_array = N(k,l) -> matrix with elements the number of edges with k 1s and size ls """
    hypergraph,opinions_hypergraph,op_dict, m_array, rho_array =iterations(hypergraph,op_dict,num_it,gamma,p,alpha) 
    num_array=np.full((len(op_dict),len(op_dict)),-1) #Matrix with N(k,l) = number of edges with k 1s and size l
    
    num_species=len(hypergraph) 
    
    size_species=np.zeros(len(hypergraph))
    ratio_ones=0
    for i in range(len(hypergraph)):
        size_species[i]=len(hypergraph[i])

    for i in op_dict:
        if op_dict[i]==1:
            ratio_ones+=1/len(op_dict)
    
    for i in size_species:
        num_array[:,int(i)]=0
    
    for i in range(len(hypergraph)):
        for j in range(len(hypergraph[i])+1):
            if opinions_hypergraph[i].count(1)==j:
                num_array[j,len(hypergraph[i])]+=1/np.count_nonzero(size_species == len(hypergraph[i]))
    return num_species, size_species, ratio_ones, m_array, rho_array, num_array

def statistics_calc_multi(num_simulations,hypergraph, opinions_hypergraph, op_dict,num_it,gamma,k,l,p,alpha):
    """For an initial hypergraph, it calculates arrays up to :
        num_species = number of edges for each simulation
        size_speces = array with the size of each edge for each simulation
        ratio_ones = the fraction of nodes with ones in the whole hypergraph for each simulation 
        m_multi_array = list of m_array for every single simulation
        rho_multi_array = list of rho_array for every single simulation
        num_array_total = list of num_array for every single simulation"""
    size_array=[]
    m_multi_array=[]
    rho_multi_array=[]
    num_sp_array=np.zeros(num_simulations)
    ratio_ones_array=np.zeros(num_simulations)
    num_array_tot=np.zeros(num_simulations)
    for i in range(num_simulations):
        hypergraph_init=copy.deepcopy(hypergraph) #after each simulation, we want the initial hypergraph to be the given one
        opinions_init=copy.deepcopy(opinions_hypergraph)
        op_dict_init=copy.deepcopy(op_dict)
        num_sp_array[i], size_var, ratio_ones_array[i], m_array, rho_array, num_array=statistics_calc(hypergraph_init, opinions_init, op_dict_init,num_it,gamma,p,alpha)
        num_array_tot[i]=num_array[k,l]
        m_multi_array.append(m_array)
        rho_multi_array.append(rho_array)
        size_array.extend(size_var)   
    return num_sp_array,size_array,ratio_ones_array, m_multi_array, rho_multi_array,num_array_tot

def m_rho_iter(hypergraph,op_dict,num_it,gamma,p,alpha):
    "Calculates magnetization and density of inactive edges FOR ONE TRAJECTORY"
    hypergraph,opinions_hypergraph,op_dict, m_array, rho_array =iterations(hypergraph,op_dict,num_it,gamma,p,alpha)     
    return m_array, rho_array

def m_rho_iter_traj(num_simulations,hypergraph, op_dict,num_it,gamma,p,alpha):
    """"Output: List of list with trajectories of magnetization and density of inactive edges"""
    m_multi_array=[]
    rho_multi_array=[]
    for i in range(num_simulations):
        hypergraph_init=copy.deepcopy(hypergraph) #after each simulation, we want the initial hypergraph to be the given one
        op_dict_init=copy.deepcopy(op_dict)
        m_array, rho_array=m_rho_iter(hypergraph_init,op_dict_init,num_it,gamma,p,alpha)
        m_multi_array.append(m_array)
        rho_multi_array.append(rho_array)
    return m_multi_array, rho_multi_array

def impact_edges_evolution(hypergraph,op_dict,num_it,gamma,p,alpha):
    """Calculates the evolution of the impact edges (most initial consensus edges S-UNIFORM)
    Main Output: Returns time array whose each element is an array with the fraction of 1s of each impactful edge  """
    opinions_hypergraph=assign_hyper_opinions(op_dict,hypergraph)
    nkl=init_impact_edges(hypergraph,opinions_hypergraph)

    m_array=[]
    rho_array=[]
    op_nkl=assign_hyper_opinions(op_dict,nkl) #depicts the initial impactful edges into opinion space
    fr_it=[]
             
    for i in range(num_it):
        # op_nkl_array[i]=op_nkl
        fr_edges=[] #array of fraction of 1s of each edge at a time
        for j in range(len(op_nkl)):
            if op_nkl[j]:
                fraction=count_opinions(op_nkl[j])
          
                fr_edges.append(fraction)
            elif not op_nkl[j]:
                fr_edges.append(-10)
            
        fr_it.append(fr_edges)
        m, rho= magnetization_density(opinions_hypergraph)
        m_array.append(m)
        rho_array.append(rho)
        
        id_edge=randrange(len(hypergraph)) #chooses a random hyperedge of the hypergraph
        hypergraph, opinions_hypergraph,op_dict=decide(id_edge,hypergraph,opinions_hypergraph,op_dict,gamma,p,alpha) #split or influence
        op_nkl=assign_hyper_opinions(op_dict,nkl)
        
        hypergraph_set=[set(i) for i in hypergraph]
        for j in range(len(nkl)): #if impact edges have split, then their evolution stops
            if not set(nkl[j]) in hypergraph_set:
                op_nkl[j]=[]
        
        if all([len(set(num)) == 1 for num in opinions_hypergraph]):
            print("Reached Equilibrium at iteration {:d}" .format(i))
            time_equilibrium=i
            break
        else:
            time_equilibrium=num_it-1
            
    return hypergraph,opinions_hypergraph,op_dict, m_array, rho_array, fr_it, time_equilibrium
        
