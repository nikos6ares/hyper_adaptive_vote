# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 14:11:41 2021

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



def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        

def minority(op_hyperedge):
    """"Calculates which opinion is a minority in the edge. 
    Used for influence to apply the update rules on the minority edges"""
    i=0
    if op_hyperedge.count(1)>op_hyperedge.count(0):
        i=0
    elif op_hyperedge.count(1)<op_hyperedge.count(0):
        i=1
    elif op_hyperedge.count(1)==op_hyperedge.count(0):
        i=random.randint(0,1)
    return i

def assign_opinions(N):
    """"Creates an array with random opinions. 
    Each index corresponds to the node of the network => Assigns an opinion to every node"""
    nodes=list(range(N))
    # opinions=np.random.randint(2, size=N).tolist() #INSTEAD OF RANDOM DO DETERMINISTICALLY TO REDUCE NOISE
    # np.random.seed(123)

    # opinions=(np.random.choice([0, 1], size=N, p=[.43, .57])).tolist()
    opinions=(np.random.choice([0, 1], size=N, p=[.5, .5])).tolist()

    print(opinions)
    op_dict=dict(zip(nodes,opinions))
    return op_dict


def assign_opinions_asymmetry(N,asymmetry):
    """"Creates an array with random opinions and specific asymmetry. 
    Each index corresponds to the node of the network => Assigns an opinion to every node"""
    nodes=list(range(N))
    asymmetry_num=float(asymmetry)/100
    # np.random.seed(149)
    opinions=(np.random.choice([0, 1], size=N, p=[1-asymmetry_num, asymmetry_num])).tolist()

    print(opinions)
    op_dict=dict(zip(nodes,opinions))
    return op_dict

def assign_hyper_opinions(op_dict,hypergraph):
    """Depicts the NODES list into the opinions of the nodes instead of the ID of each node"""
    opinions_hyper=[]
    for i in range(len(hypergraph)):
        opinions_hyper.extend([[op_dict[key] for key in hypergraph[i]]]) #Creates superlist, with lists of the opinions of the nodes on each edge
    return opinions_hyper

def assign_hyperedge_op(op_dict,edge):
    """Depicts the HYPEREDGE list into the opinions of the nodes instead of the ID of each node"""

    opinions_hyper=[0]*len(edge)
    for i in range(len(edge)):
        opinions_hyper=[op_dict[key] for key in edge] #Creates superlist, with lists of the opinions of the nodes on each edge
    return opinions_hyper    

def remove_dublicate(seq):
    """"Removes nodes that are twice in an edge"""
    seen = set() 
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def count_opinions(opinions_hyperedge):
    """"Counts the fraction of 1 opinions on the hyperedge"""
    edge_fr=opinions_hyperedge.count(1)/len(opinions_hyperedge)
    return edge_fr

def node_count_opinions(op_dict):
    """Counts the fraction of 1 opinions on hypergraph - node based"""
    f=0
    
    for i in range(len(op_dict)):
        if op_dict[i]==1:
            f+=1/len(op_dict)
    return f

def magnetization_density(opinions_hyper):
    """"Calculates the magnetization and the density of active edges of the hypergraph: 
        m= number_ones-number_zeros 
        rho = number_uniform_edges"""
    m=0
    rho=0
    normal=0
    for i in range(len(opinions_hyper)):
        m+=(opinions_hyper[i].count(1)-opinions_hyper[i].count(0))
        normal+=len(opinions_hyper[i])
        if len(set(opinions_hyper[i])) == 1:
            rho+=1/len(opinions_hyper)
    return m/normal, rho

def node_magnet_density(opinions_hyper,op_dict):
    """"Calculates the node-based magnetization and the density of active edges of the hypergraph: 
        m= number_ones-number_zeros 
        rho = number_uniform_edges"""
    m=0
    rho=0
    for i in range(len(opinions_hyper)):
        if len(set(opinions_hyper[i])) == 1:
            rho+=1/len(opinions_hyper)
            
    for i in range(len(op_dict)):
        m+=(2*op_dict[i]-1)
        
    return m/len(op_dict), rho

def number_id(k,l,opinions_hyper):
    """Calculates N(k,l) for a given configuration. N(k,l)=number of edges with k 1s opinions, l size."""
    number_id=[]
    for i in range(len(opinions_hyper)):
        if opinions_hyper[i].count(1)==k and len(opinions_hyper[i])==l:
            number_id.append(i)
    return number_id

def Nkl_calculator(hypergraph,opinions_hyper):
    "Calculates N(k,S) for S-Uniform hypergraph"
    print(int((len(hypergraph[0]))))
    k=np.zeros(int((len(hypergraph[0])))+1)
    for i in range(len(hypergraph)):
        k[opinions_hyper[i].count(1)]+=1
    return k

def moments(array):
    "Calculates the mean and the standard deviation of an array"
    mean=0 #mean of final absolute magnetizations
    var=0 #variance of final absolute magnetizations
    stdev=0 #sqrt of var
    for j in range(len(array)):
        mean+=array[j]/len(array)
    for j in range(len(array)):
        var+=(array[j]-mean)**2/len(array)
    stdev=np.sqrt(var)
    return mean, stdev

def moments_none(array):
    "Calculates the mean and the standard deviation of an array. Considers NONE as 0s"
    mean=0 #mean of final absolute magnetizations
    var=0 #variance of final absolute magnetizations
    stdev=0 #sqrt of var
    for j in range(len(array)):
        if array[j]:
            mean+=array[j]/sum(x is not None for x in array)
    for j in range(len(array)):
        if array[j]:
            var+=(array[j]-mean)**2/sum(x is not None for x in array)
    stdev=np.sqrt(var)
    return mean, stdev

def moments_filter(array,initial):
    "Calculates the mean and the standard deviation of an array. Considers only trajectories with m>m_0"
    mean=0 #mean of final absolute magnetizations
    var=0 #variance of final absolute magnetizations
    stdev=0 #sqrt of var
    length=len(array)
    array=[item for item in array if item >= initial]
    for j in range(len(array)):            
        mean+=array[j]/len(array)
    for j in range(len(array)):
        var+=(array[j]-mean)**2/len(array)
    stdev=np.sqrt(var)
    return mean, stdev


def init_impact_edges(hypergraph,opinions_hyper):
    """Calculates which edges at the initial time have surplus opinions and returns the ones with the most surplus:
        most surplus: ind_max
        the two most surplus: ind_2max
        the three most surplus: ind_3max"""
    k_surplus=[0]*len(hypergraph)
    for i in range(len(hypergraph)):
        minor=minority(opinions_hyper[i])
        major=abs(1-minor)
        if opinions_hyper[i].count(major)>len(hypergraph[i])/2:
            k_surplus[i]=opinions_hyper[i].count(major)-len(hypergraph[i])/2
            
    ind_max=[i for i, j in enumerate(k_surplus) if j == max(k_surplus)]
    # ind_2max=[i for i, j in enumerate(k_surplus) if j == sorted(set(k_surplus))[-2] and sorted(set(k_surplus))[-2]>0]
    # ind_max.extend(ind_2max)
    # ind_3max=[i for i, j in enumerate(k_surplus) if j == sorted(set(k_surplus))[-3] and sorted(set(k_surplus))[-3]>0]
    # ind_max.extend(ind_3max)
    impact_edges=[]
    for i in range(len(ind_max)):
        impact_edges.append(hypergraph[ind_max[i]])
        
    return impact_edges

def prob_dist_size(N,beta):
    """Probability distribution for the size of the edges for the Heterogeneous Mean Field Case"""
    size=int(2+np.floor(np.random.exponential(scale=beta, size=None)))
    while size>N:
        size=int(2+np.floor(np.random.exponential(scale=beta, size=None)))
    return size
    
def group_opinions(op_dict):
    """Groups indices of nodes to two arrays with 1 opinions or 0 opinions"""
    nodes_1=[]
    nodes_0=[]
    for i in range(len(op_dict)):
        if op_dict[i]==1:
            nodes_1.append(i)
        else:
            nodes_0.append(i)
    return nodes_1, nodes_0