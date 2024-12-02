# An Adaptive Voter Model on Hypergraphs

## List of scripts and their functions:

1. **dynamics.py** : 
* Function: Simulates adaptation, rewire, influence, split, merge and re-scaled γ (for discordance function)
* Used: In all of the scripts
2. **iterating.py** :
* Function: Iterates the dynamics for many trajectories or configurations
* Used: In suniform.py and heterogeneous.py
3. **hypergraph.py**: 
* Function: Creates initial hypergraph: either uniform or heterogeneous with exponential size distribution
* Used: In suniform.py, heterogeneous.py and iterating.py
4. **tools.py**:
* Function: Small functions that define functions/actions in the other scripts: e.g. assign_opinions_asymmetry.
* Used: In all of the scripts
5. **suniform.py**:
* Function: Collection of functions to generate data of studying S-uniform hypergraph: e.g. abs m vs gamma, symmetry restoration vs N 
6. **plot_suniform.py**:
* Function: Collection of functions to plot generated data of studying S-uniform hypergraph
7. **hmf.py**:
* Function: Simulates the HMF rules (dynamics) and generates data of studying hmf including Kramer's escape rate calculations
8. **plot_hmf.py**:
* Function: Plots generated data of studying hmf
9. **heterogeneous.py**:
* Function: Collection of functions to generate data of studying heterogeneous initial hypergraph: e.g. abs m vs gamma,  abs m vs beta 
10. **plot.heterogeneous.py**:
* Function: Collection of functions to plot generated data of studying heterogeneous initial Hypergraphs
11. **hmf_pool.py**:
* Function: Same with hmf.py but used for EULER CLUSTER to generate data : magnetization/time of convergence vs a quantity, escape rate vs initial magnetization, magnetization/time of convergenve vs alpha (discordance function). Properly parallelized using pooling starmap. The functions that are used for pooling start with "pool" in their name.
12. **heterogeneous_pool.py**:
* Function: Same with heterogeneous.py but used for EULER CLUSTER to generate data: abs magnetization vs a quantity (beta or gamma). Properly parallelized using pooling starmap. The functions that are used for pooling start with "pool" in their name.

## Execution:
* Use: **suniform.py** for S-uniform, **hmf.py** for hmf, **heterogeneous.py** for heterogeneous initial configuration.
* At the end of these scripts there are many commented lines with short description what they do. Uncomment/comment depending on the desired output. If data are loaded instead of generated, transfer the data to array to define inputs  of plot functions.
* Parameters are given below the function definitions.

## Possible Adjustments:
* To choose between edge-based magnetization vs node-based magnetization: Uncomment/comment lines 32/33 at iterating.py in iterations function
* To switch on/off rewire and merge: Uncomment/comment lines 112 and 257 at dynamics.py in split and two-split functions
* To study the effect of α at the discordance function: Choose a different value for alpha in the parameters of the executable functions
* To vary gamma or beta at heterogeneous.py comment/uncomment commented/uncommented lines in heterogeneous.py at m_vs_gamma_multi_config function.
* To study a specific configuration in S-uniform model, comment/uncomment lines of random.seed in hypergraph.py at create_hypergraph function AND random.seed in tools.py at assign_opinions_asymmetry function.

## Current Features:
For the possible plots see description of functions in: plot_hmf.py, plot_suniform.py, plot_heterogeneous.py.

## TODO:
* Add menu to choose: 1. output of hmf.py, suniform.py, heterogeneous.py, 2. to generate data or load data







