# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:59:59 2021

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
from scipy.optimize import curve_fit



from tools import group_opinions,minority, assign_hyper_opinions, count_opinions, remove_dublicate
    

def alpha_gamma(alpha,gamma,edge):
    """Rescaling the given gamma depending on alpha.
    Used before the Decision function"""
    gamma=gamma*(len(edge))**(alpha-1)
    return gamma
    

def influence(id_edge,hypergraph, opinions_hyper,op_dict): 
    """"Implements the influence step.
    Updates the opinions array and the opinions edge on the minority vertices"""
    op_hyperedge=opinions_hyper[id_edge]
    hyperedge=hypergraph[id_edge]
    minor=minority(op_hyperedge) #value of the minority opinion
    prob=op_hyperedge.count(minor)/len(op_hyperedge) #probability staying the same
    indices = [i for i, x in enumerate(op_hyperedge) if x == minor] #which nodes are in the minority
    for i in range(len(indices)):
        if random.random()<=1-prob: #with probability 1-p, we change the opinion
            node=hyperedge[indices[i]] #ID of the node with minority opinion
            op_dict[node]=op_dict[node]*(-1)+1
            opinions_hyper=assign_hyper_opinions(op_dict,hypergraph)
    return opinions_hyper, op_dict


def merge(hypergraph,opinions_hyper, op_dict, gamma):
    """"Calculates the hypergraph after splitted edges have merged with random edges
    The random edges must have a "tolerated" majority opinion equal with the opinion of the split edge
    Output: The new hypergraph with its opinion depiction"""
    aux = copy.deepcopy(hypergraph) #makes a copy of hypergraph
    aux.remove(hypergraph[-1]) #removes the last 2 edges which were split
    aux.remove(hypergraph[-2])
    random.shuffle(aux) #so that we choose randomly an edge to attach the split edge
    # aux.sort(key=len) #so that we start attaching firstly the ones with smallest size
    
    op_aux=assign_hyper_opinions(op_dict, aux)
    edges_temp=[hypergraph[-1],hypergraph[-2]] #split edges array
    edges=copy.deepcopy(edges_temp) 
    op_edges=assign_hyper_opinions(op_dict,edges)
    
    hypergraph.remove(edges[0]) #removes the last 2 edges which were split from the final hypergraph
    hypergraph.remove(edges[1])
    

    for i in range(2): #going through the split edges of hypergraph
        
        for j in (range(len(aux))): #going through the small edges of aux
                
            edge_fr=count_opinions(op_aux[j]) #counts the fraction of 1s on each edge of aux

            if (edge_fr< gamma) and op_edges[i][0]==0: #if "tolerated" majority of edge is 0 and split edge = 0

                hypergraph.remove(aux[j]) #removes the chosen random edge

                    
                edges[i]=aux[j]+edges[i] #merge the random edge with split edge
                
                break #once we merge two edges we stop the loop
            elif edge_fr>1-gamma and op_edges[i][0]==1: #similarly for majority = 1 
                hypergraph.remove(aux[j]) #removes the chosen random edge

                edges[i]=aux[j]+edges[i] #merge the random edge with split edge
               
            #if edge_fr=gamma then nothibng happens
                break
                
    edges[0]=remove_dublicate(edges[0]) #remove double nodes from the edge
    edges[1]=remove_dublicate(edges[1]) #remove double nodes from the edge
    hypergraph.append(edges[0]) #attach new merged edge to hypergraph
    hypergraph.append(edges[1]) #attach new merged edge to hypergraph
    hypergraph=list(unique_everseen(hypergraph, key=frozenset)) #similar to remove_dublicate but for nested lists
    opinions_hyper=assign_hyper_opinions(op_dict, hypergraph) 

    return hypergraph, opinions_hyper


def split(id_edge, hypergraph,opinions_hyper,op_dict,gamma):
    """Splits a hyperedge into two hyperedges with the same kind of opinions"""
    
    hyperedge=hypergraph[id_edge] 
    splt_edges=[[num for num in hyperedge if op_dict[num]==o] for o in [0,1]] #splits the ID edge into 2 edges with same opinions
    hypergraph=[num for num in hypergraph if num!=hyperedge] #deletes the initial edge from hypergraph
    hyperedge=splt_edges #updates the split edge
    hypergraph.extend(hyperedge) #adds the new edges in hypergraph
    opinions_hyper=assign_hyper_opinions(op_dict,hypergraph) #creates an opinion depiction of the new hypergraph
    
    #Following line - commented : merge off
    # Set γ=0.5 in merge = split edges merge with edges whose majority is the same 
    hypergraph, opinions_hyper=merge(hypergraph,opinions_hyper, op_dict, 0.5)
    

    return hypergraph,opinions_hyper

def decide(id_edge,hypergraph,opinions_hyper,op_dict,gamma,p,alpha):
    """"Decides whether influence or splitting by calculating fraction of opinions 1 on hyperedge"""
    op_edge=opinions_hyper[id_edge] #choosing edge
    edge=hypergraph[id_edge]
    gamma=alpha_gamma(alpha,gamma,edge)
    edge_fr=count_opinions(op_edge) #counting fraction of 1 
    if len(op_edge)!=2:
        if gamma>0.5:
            gamma_cor=1-gamma
        else:
            gamma_cor=gamma
        if edge_fr<= gamma_cor or edge_fr>=1-gamma_cor:
            opinions_hyper, op_dict=influence(id_edge,hypergraph, opinions_hyper,op_dict)
        else: # gamma<=fraction<=1-gamma => splitting regime
            hypergraph,opinions_hyper=split(id_edge, hypergraph,opinions_hyper,op_dict,gamma)
    elif len(op_edge)==2:
        hypergraph,opinions_hyper,op_dict=two_edge_decide(id_edge,hypergraph,opinions_hyper,op_dict,gamma,p)
    return hypergraph, opinions_hyper,op_dict

def two_edge_decide(id_edge,hypergraph,opinions_hyper,op_dict,gamma,p):
    """"Classical Adaptive Voter Model (2-edges) decide function"""
    op_edge=opinions_hyper[id_edge]
    edge_fr=count_opinions(op_edge)
    if (edge_fr!=0 and edge_fr!=1):
        if random.random()<=p: #p is probability of rewiring
            hypergraph,opinions_hyper=two_split(id_edge, hypergraph,opinions_hyper,op_dict,gamma)
        else:
            node=random.randrange(0,2)
            op_dict[hypergraph[id_edge][node]]=op_dict[hypergraph[id_edge][node]]*(-1)+1
            opinions_hyper=assign_hyper_opinions(op_dict,hypergraph)
    return hypergraph,opinions_hyper,op_dict



def two_merge(hypergraph,opinions_hyper, op_dict, gamma):
    """"TODO: DELETE THIS FUNCTION
    FOR 2 EDGES Calculates the hypergraph after splitted edges have merged with random edges
    The random edges must have a "tolerated" majority opinion equal with the opinion of the split edge
    Output: The new hypergraph with its opinion depiction"""
    aux = copy.deepcopy(hypergraph) #makes a copy of hypergraph
    aux.remove(hypergraph[-1]) #removes the last 2 edges which were split
    aux.remove(hypergraph[-2])
    random.shuffle(aux) #so that we choose randomly an edge to attach the split edge
    aux.sort(key=len) #so that we start attaching firstly the ones with smallest size
    
    op_aux=assign_hyper_opinions(op_dict, aux)
    edges_temp=[hypergraph[-1],hypergraph[-2]] #split edges array
    edges=copy.deepcopy(edges_temp) 
    op_edges=assign_hyper_opinions(op_dict,edges)
    
    hypergraph.remove(edges[0]) #removes the last 2 edges which were split from the final hypergraph
    hypergraph.remove(edges[1])
    

    for i in range(2): #going through the split edges of hypergraph
        
        for j in (range(len(aux))): #going through the small edges of aux
            if len(op_aux[j])<2:
                edge_fr=count_opinions(op_aux[j]) #counts the fraction of 1s on each edge of aux

                if (edge_fr< gamma) and op_edges[i][0]==0: #if "tolerated" majority of edge is 0 and split edge = 0
    
                    hypergraph.remove(aux[j]) #removes the chosen random edge

                    edges[i]=aux[j]+edges[i] #merge the random edge with split edge
                    
                    break #once we merge two edges we stop the loop
                    
                elif edge_fr>1-gamma and op_edges[i][0]==1: #similarly for majority = 1 
                    hypergraph.remove(aux[j]) #removes the chosen random edge

                    edges[i]=aux[j]+edges[i] #merge the random edge with split edge
                   
                    break
                    
    edges[0]=remove_dublicate(edges[0]) #remove double nodes from the edge
    edges[1]=remove_dublicate(edges[1]) #remove double nodes from the edge

    hypergraph.append(edges[0]) #attach new merged edge to hypergraph
    hypergraph.append(edges[1]) #attach new merged edge to hypergraph
    hypergraph=list(unique_everseen(hypergraph, key=frozenset)) #similar to remove_dublicate but for nested lists
    opinions_hyper=assign_hyper_opinions(op_dict, hypergraph) 

    return hypergraph, opinions_hyper

def two_rewire(hypergraph,opinions_hyper, op_dict):
    "Simulates rewire mechanism"
    nodes_other=np.arange(len(op_dict))
    split_edge1=copy.deepcopy(hypergraph[-1]) #the split edges (one single node each)
    split_edge2=copy.deepcopy(hypergraph[-2])
    hypergraph.remove(hypergraph[-1])
    hypergraph.remove(hypergraph[-1])
    
    

    nodes_other=nodes_other[nodes_other != split_edge1]
    nodes_other=nodes_other[nodes_other != split_edge2]

    nodes_1=[]
    nodes_0=[]
    for i in range(len(nodes_other)): #groups the remaining nodes
        if op_dict[nodes_other[i]]==1:
            nodes_1.append(nodes_other[i])
        else:
            nodes_0.append(nodes_other[i])
    
    if split_edge1==0: #if edge1 is 0 then edge2 is 1
        if len(nodes_0)!=0 and len(nodes_1)!=0:
            split_edge1.append(random.choice(nodes_0)) #rewire node in edge1 with remaining random node with opinion 0 
            split_edge2.append(random.choice(nodes_1))
        elif len(nodes_0)!=0 and len(nodes_1)==0: #if there are no nodes with opinion 1 then leave node unrewired
            split_edge1.append(random.choice(nodes_0))
        elif len(nodes_0)==0 and len(nodes_1)!=0:
            split_edge2.append(random.choice(nodes_1))
    else:
        if len(nodes_0)!=0 and len(nodes_1)!=0:
            split_edge1.append(random.choice(nodes_1))
            split_edge2.append(random.choice(nodes_0))
        elif len(nodes_0)!=0 and len(nodes_1)==0:
            split_edge2.append(random.choice(nodes_0))
        elif len(nodes_0)==0 and len(nodes_1)!=0:
            split_edge1.append(random.choice(nodes_1))
   
    hypergraph.append(split_edge1)
    hypergraph.append(split_edge2)
    hypergraph=list(unique_everseen(hypergraph, key=frozenset))
    opinions_hyper=assign_hyper_opinions(op_dict, hypergraph) 
    return hypergraph, opinions_hyper
    
def two_split(id_edge, hypergraph,opinions_hyper,op_dict,gamma):
    """Splits a hyperedge into two hyperedges with the same kind of opinions"""
    
    hyperedge=hypergraph[id_edge] 
    splt_edges=[[num for num in hyperedge if op_dict[num]==o] for o in [0,1]] #splits the ID edge into 2 edges with same opinions
    hypergraph=[num for num in hypergraph if num!=hyperedge] #deletes the initial edge from hypergraph
    hyperedge=splt_edges #updates the split edge
    hypergraph.extend(hyperedge) #adds the new edges in hypergraph
    opinions_hyper=assign_hyper_opinions(op_dict,hypergraph) #creates an opinion depiction of the new hypergraph
    #merge mode: ON
    # Set γ=0.5 in merge = split edges merge with edges whose majority is the same 
    hypergraph, opinions_hyper=two_rewire(hypergraph,opinions_hyper, op_dict)

    return hypergraph,opinions_hyper

