#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import scipy
from scipy import stats
import pandas as pd
import csv
from scipy.stats import unitary_group
########################################################################## Parameters
## number of nodes
n = 100
## runtime for the simulation
runtime = n*210
stabtime = n*10
## sample over runs
RUN = 10

##################################################################### Graph related functions
def get_neighbours(adj_mat,n):
    ## function to return list of neighbours of all nodes
    neighbours = []
    for node in range(0,n):
        for i in range(0,n):
            if (adj_mat[node,i] == 1):
                neighbours.append(i)
    return neighbours

def make_graph(n, seedling, mean_k):
    ## Choose p to target a desired average degree: E[k] = p*(n-1)
    # p = mean_k/(n-1)
    # G = nx.erdos_renyi_graph(n=n, p=p, seed=seedling)
    ## small world network 
    p = 0.1
    G = nx.watts_strogatz_graph(n, k=mean_k, p=p)
    ## list of degree of nodes 0 to n
    links = np.array([d for _, d in G.degree()]).astype('int')    
    adj_mat = nx.to_numpy_array(G).astype('int')
        ## cummulative sum of the edges, used to access the state vector elements
    cum_links = np.cumsum(links).astype('int')
    N = int(cum_links[n-1])                  ## this is twice the total number of edges in directed-arc space
    cum_links = np.insert(cum_links,0,0)
    print("twice the total number of edges: ",N)
    print("average degree: ", np.mean(links))
    ## get list of neighbours of all nodes
    neighbour_list = np.array(get_neighbours(adj_mat, n))
    # print(neighbour_list)
    return N, links, cum_links, neighbour_list

############################################################################# Coin functions
def create_grover(k):
    mat = np.empty((k,k))
    mat.fill(2)
    for i in range(0,k):
        mat[i][i] = 2-k
    mat /= k
    return mat
            
def create_fourier(k):
    mat = np.ones((k,k),dtype=complex)
    for i in range(1,k):
        for j in range(1,k):
            mat[i][j] = np.exp(2j*np.pi*i*j/k)
    mat /= np.sqrt(k)
    return mat


def mint_coin(n, links):
    ## Grover coins for each node
    # my_coins = []
    # for i in range(0,n):
    #     my_coins.append(create_grover(links[i]))

    ## Fourier coins for each node
    my_coins = []
    for i in range(0,n):
        my_coins.append(create_fourier(links[i]))
    
    Coin_AR = scipy.linalg.block_diag(*my_coins)
    return Coin_AR
    
############################################################################# Shift operator
def make_shift(n, N, links, cum_links, neighbour_list):    
    ## Arc-reveral model        
    Shift_AR = np.zeros((N,N))
    for node in range(0,n):
        for j in range(cum_links[node],cum_links[node]+links[node]):
            node_neighbour = int(neighbour_list[j])
            for k in range(cum_links[node_neighbour],cum_links[node_neighbour]+links[node_neighbour]):
                if(neighbour_list[k] == node):
                    Shift_AR[j,k] = 1
    return Shift_AR

########################################################################### Unitary as product of shift and coin
def get_U(n, N, links, cum_links, neighbour_list):
    shift = make_shift(n, N, links, cum_links, neighbour_list)
    coin = mint_coin(n, links)
    
    U = np.matmul(shift,coin)
    F_list = []
    return shift, coin, U, F_list

################################################################################################
## initialization at a particular 'node'
def initialize(N, k, cummulative, degree):
    state = np.zeros(N, dtype=np.complex128)
    if degree > 0:
        for i in range(cummulative, cummulative+degree):
            state[i] = 1/np.sqrt(degree)
    print("state initialized at node {}, degree = {}.".format(k, degree))
    return state

## to get probability amplitudes at each node
def magn(state, n, links, cum_links):
    magnit = np.zeros(n)
    for i in range(0,n):
        for j in range(0,links[i]):
            magnit[i] += abs(state[cum_links[i]+j])**2
    return magnit


#######################################################################################
def check_off_diag_sum(N, F_list):
    pos = np.arange(0,N)
    not_trace = np.zeros(N)
    J = np.ones((N,N))
    for i in range(0,N):
        J[i,i] = 0
    for i in range(0, len(F_list)):
        not_trace[i] = np.trace(np.matmul(J, F_list[i]))
    not_trace = np.around(not_trace, 2)
    plt.plot(pos, not_trace, '.')
    plt.xlabel("F_r")
    plt.ylabel("off diagonal sum")
    plt.show()


####################################################################################### EE stats
def count_rouge(n, magnit, P_th, runtime, stabtime):
    EE = np.zeros(n)
    avg_recur = np.zeros(n)
    ## to get recurrence times at, say node 16
    recur1 = []
    node1 = 16
    max1 = P_th[node1]
    
    T = runtime-stabtime
    for node in range(0,n):
        timestamps = [stabtime]
        for t in range(stabtime, runtime):
            if (magnit[t, node] > P_th[node]):
                EE[node] += 1
                timestamps.append(t)
                if (node == 16):
                    recur1.append(t-timestamps[-2])
        if(timestamps[-1] == 0):
            timestamps[-1] = runtime
        if EE[node] > 0:
            avg_recur[node] = (timestamps[-1]-timestamps[0])/EE[node]
        else:
            avg_recur[node] = 0
    EE = EE*100/(T)
    return EE, avg_recur, recur1


def do_stats(run, n, runtime, stabtime, g, m=2.0):
    means = np.zeros(n)
    stds = np.zeros(n)
    P_th = np.zeros(n)
    EE = np.zeros(n)
    ## Match filename written by dtqw
    data_dtqw = np.loadtxt(f"Mstd_magnitudes_{n}nodes_meank{g}.dat")
    for node in range(0,n):
        means[node] = np.mean(data_dtqw[stabtime:runtime, node])
        stds[node] = np.std(data_dtqw[stabtime:runtime, node])
    P_th = means + m*stds
    EE, avg_recur, recurtimes = count_rouge(n, data_dtqw, P_th, runtime, stabtime)
    return means, stds, EE, avg_recur, recurtimes
    

##################################################################################### Initialise and evolve the system here for DTQW
def dtqw(n, N, links, cum_links, neighbour_list, g):
    Shift_AR, Coin_AR, U, F_list = get_U(n, N, links, cum_links, neighbour_list)
    state = np.zeros(N, dtype=np.complex128)
    magnitude = np.zeros(n)
    EE_runAvg = np.zeros(n)
    mean_runAvg = np.zeros(n)
    std_runAvg = np.zeros(n)
    recur_avg = np.zeros(n)
    recurtimes = [] 

    ## precompute nodes with degree > 0 to avoid invalid initialization
    nodes_with_edges = np.where(links > 0)[0]
    if nodes_with_edges.size == 0:
        raise ValueError(f"Generated ER graph has no edges; increase p or target_avg_degree for meank{g}.")

    for run in range(0,RUN):
        ## initialize at some node k with nonzero degree
        k = int(np.random.choice(nodes_with_edges))
        state = initialize(N, k, cum_links[k], links[k])
        ## pre-allocate magnitude data storage for better performance
        magn_data = np.zeros((runtime, n + 1))  # +1 for sum column
        magnitude = magn(state, n, links, cum_links)
        magn_data[0, :n] = magnitude
        magn_data[0, n] = np.sum(magnitude)
        
        ## evolve the system for from 1 to runtime
        for t in range(1, runtime):
            state = np.matmul(U, state)
            magnitude = magn(state, n, links, cum_links)
            magn_data[t, :n] = magnitude
            magn_data[t, n] = np.sum(magnitude)
        
        ## write magnitude data efficiently using numpy savetxt
        np.savetxt(f"Mstd_magnitudes_{n}nodes_meank{g}.dat", magn_data, fmt='%.6f', delimiter=' ')

        mean_, std_, EE_, recur_avg_, recurtime_ = do_stats(run, n, runtime, stabtime, g)
        mean_runAvg += mean_
        std_runAvg += std_
        EE_runAvg += EE_
        recur_avg += recur_avg_
        recurtimes = np.concatenate((recurtimes, recurtime_))
    ## finally return averages (over RUN) of means, stds, and EE_counts
    return mean_runAvg/RUN, std_runAvg/RUN, EE_runAvg/RUN, recur_avg/RUN, recurtimes


##############################################################################################  Run over various graphs
mean_k_vals = [2,4,6,8,10,12,14,16,18,20]
for g in mean_k_vals:
    rnd = 576711
    N, links, cum_links, neighbour_list = make_graph(n, rnd, g)
    mean_runAvg, std_runAvg, EE_runAvg, recur_avg, recurtimes = dtqw(n, N, links, cum_links, neighbour_list, g)
    # np.savetxt(f"mean_runAvg_n{n}_meank{g}_grover.csv", mean_runAvg, fmt='%.6f', delimiter=',', newline='\n')
    # np.savetxt(f"std_runAvg_n{n}_meank{g}_grover.csv", std_runAvg, fmt='%.6f', delimiter=',', newline='\n')
    # np.savetxt(f"EE_runAvg_n{n}_meank{g}_grover.csv", EE_runAvg, fmt='%.6f', delimiter=',', newline='\n')
    # np.savetxt(f"recur_avg_k_n{n}_meank{g}_grover.csv", recur_avg, fmt='%.6f', delimiter=',', newline='\n')
    # np.savetxt(f"recurtimes_n{n}_meank{g}_grover.csv", recurtimes, fmt='%.6f', delimiter=',', newline='\n')
    # np.savetxt(f"links_n{n}_meank{g}_grover.csv", links, fmt='%.6f', delimiter=',', newline='\n')

    ## uncomment the following for Watts-Strogatz graph
    np.savetxt(f"mean_runAvg_n{n}_meank{g}.csv", mean_runAvg, fmt='%.6f', delimiter=',', newline='\n')
    np.savetxt(f"links_n{n}_meank{g}.csv", links, fmt='%.6f', delimiter=',', newline='\n')
    np.savetxt(f"std_runAvg_n{n}_meank{g}.csv", std_runAvg, fmt='%.6f', delimiter=',', newline='\n')
    np.savetxt(f"ee_runAvg_n{n}_meank{g}.csv", EE_runAvg, fmt='%.6f', delimiter=',', newline='\n')
    np.savetxt(f"recur_avg_k_n{n}_meank{g}.csv", recur_avg, fmt='%.6f', delimiter=',', newline='\n')
    np.savetxt(f"recurtimes_n{n}_meank{g}.csv", recurtimes, fmt='%.6f', delimiter=',', newline='\n')


