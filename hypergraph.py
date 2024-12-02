# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:55:48 2021

@author: Nikos
"""

import numpy as np
import random
from iteration_utilities import unique_everseen
from random import randrange
from tools import prob_dist_size

def create_hypergraph(N,S,n):
    """"Create a S-uniform (all edges size S) hypergraph of N nodes with n hyperedges
    --------------------------------------------------------------------------------------
        Output: a list of hyperedges
        First stage: Making sure *all* nodes are sorted into hyperedges.
        Second stage: If first stage is completed AND a hyperedge does not have size S
        then: random nodes are filled in the empty spaces
        Third stage: For the rest of the hyperedges, S random nodes are chosen repeatedly"""
    nodes=list(range(N))
    # random.seed(128) #activate if study specific configuration

    random.shuffle(nodes)
    hypergraph=[nodes[i:i + S] for i in range(0, len(nodes), S)] #Splits nodes list into lists with size S
    if len(hypergraph[-1])!=len(hypergraph[0]):  #if there are empty slots in the last hyperedge
        nodes_reduced = [x for x in nodes if x not in hypergraph[-1]] #making sure no repetition of nodes in hyperedge
        # random.seed(120) #activate if study specific configuration
        random.shuffle(nodes_reduced)
        hypergraph[-1].extend(nodes_reduced[:(len(hypergraph[0])-len(hypergraph[-1]))]) #fills the empty slots of the last hyperedge
        
    if np.ceil(N/S)<n: #For the remaining edges, we choose repeatedly S random nodes
        rmn=n-np.ceil(N/S) #number of remaining edges
        for i in range(int(rmn)):
            
            # random.seed(125) #activate if study specific configuration
            random.shuffle(nodes)
            hypergraph.extend([nodes[:S]])
            
    hypergraph=list(unique_everseen(hypergraph, key=frozenset))

    return hypergraph

def heterogeneous_initial(N,n,beta):
    """Creates a hypergraph with a specific distribution of size of the edges
    Output: hypergraph= list of list of different sizes representing hyperedges
    Used in:  heterogeneous.py """
    hypergraph=[]
    for i in range(n):
        size=prob_dist_size(N,beta)
        edge=random.sample(range(0, N), size)
        hypergraph.append(edge)
    hypergraph=list(unique_everseen(hypergraph,key=frozenset))
    while len(hypergraph)!=n: #in case there was an identical edge
        diff=int(n-len(hypergraph)) #replaced by a new one
        for i in range(diff):
            size=prob_dist_size(N,beta)
            edge=random.sample(range(0, N), size)         
            hypergraph.append(edge)
        hypergraph=list(unique_everseen(hypergraph,key=frozenset)) #overwrites identical edges
    return hypergraph

