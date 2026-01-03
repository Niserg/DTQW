#!/usr/bin/env python
# coding: utf-8

import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
import networkx as nx
import scipy
from scipy import stats
import pandas as pd
import csv
from scipy.stats import unitary_group
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
########################################################################## Parameters
## number of nodes
n = 1000
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
    p = mean_k/(n-1)
    G = nx.erdos_renyi_graph(n=n, p=p, seed=seedling)
    ## small world network 
    # p = 0.1
    # G = nx.watts_strogatz_graph(n, k=mean_k, p=p)
    ## scale-free network via the Barabasi-Albert algorithm
    # G = nx.barabasi_albert_graph(n, m = 2, seed=seedling)
    
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
    my_coins = []
    for i in range(0,n):
        k = links[i]
        if k > 0:
            my_coins.append(create_grover(k))
        else:
            my_coins.append(np.zeros((0,0)))

    ## Fourier coins for each node
    #my_coins = []
    #for i in range(0,n):
    #    my_coins.append(create_fourier(links[i]))
    
    return my_coins
    
############################################################################# Shift operator
def make_shift_indices(n, N, links, cum_links, neighbour_list):
    ## Arc-reversal model as an index permutation
    shift_idx = np.empty(N, dtype=np.int32)
    for node in range(0,n):
        start_j = cum_links[node]
        end_j = start_j + links[node]
        for j in range(start_j, end_j):
            node_neighbour = int(neighbour_list[j])
            start_k = cum_links[node_neighbour]
            end_k = start_k + links[node_neighbour]
            # find the reverse arc position
            for k in range(start_k, end_k):
                if neighbour_list[k] == node:
                    shift_idx[j] = k
                    break
    return shift_idx

########################################################################### Unitary as product of shift and coin
def get_precomputed_ops(n, N, links, cum_links, neighbour_list):
    shift_idx = make_shift_indices(n, N, links, cum_links, neighbour_list)
    coins = mint_coin(n, links)
    return shift_idx, coins

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



####################################################################################### EE stats
def count_rouge(n, magnit, P_th, runtime, stabtime):
    EE = np.zeros(n)
    avg_recur = np.zeros(n)
    ## to get recurrence times at, say node 16
    node1 = 16
    node2 = 1
    node3 = 4
    max1 = P_th[node1]
    max2 = P_th[node2]
    max3 = P_th[node3]
    recur1 = []
    recur2 = []
    recur3 = []

    T = runtime-stabtime
    for node in range(0,n):
        timestamps = [stabtime]
        for t in range(stabtime, runtime):
            if (magnit[t, node] > P_th[node]):
                EE[node] += 1
                timestamps.append(t)
                if (node == 16):
                    recur1.append(t-timestamps[-2])
                elif (node == node2):
                    recur2.append(t-timestamps[-2])
                elif (node == node3):
                    recur3.append(t-timestamps[-2])
        if(timestamps[-1] == 0):
            timestamps[-1] = runtime
        if EE[node] > 0:
            avg_recur[node] = (timestamps[-1]-timestamps[0])/EE[node]
        else:
            avg_recur[node] = 0
    EE = EE*100/(T)
    return EE, avg_recur, recur1, recur2, recur3


def compute_stats_from_magnitudes(n, runtime, stabtime, magn_data, m=2.0):
    means = np.mean(magn_data[stabtime:runtime, :n], axis=0)
    stds = np.std(magn_data[stabtime:runtime, :n], axis=0)
    P_th = means + m*stds
    EE, avg_recur, recur1, recur2, recur3 = count_rouge(n, magn_data[:, :n], P_th, runtime, stabtime)
    return means, stds, EE, avg_recur, recur1, recur2, recur3
    

##################################################################################### Initialise and evolve the system here for DTQW
def apply_coin_step_inplace(state, n, links, cum_links, coins):
    for node in range(0, n):
        deg = links[node]
        if deg == 0:
            continue
        start = cum_links[node]
        end = start + deg
        seg = state[start:end]
        state[start:end] = coins[node].dot(seg)


def simulate_one_run(seed, n, N, links, cum_links, neighbour_list, shift_idx, coins, runtime, stabtime, g):
    rng = np.random.default_rng(seed)
    nodes_with_edges = np.where(links > 0)[0]
    if nodes_with_edges.size == 0:
        raise ValueError(f"Generated ER graph has no edges; increase p or target_avg_degree for meank{g}.")

    k = int(rng.choice(nodes_with_edges))
    state = initialize(N, k, cum_links[k], links[k])

    magn_data = np.zeros((runtime, n + 1))
    magnitude = magn(state, n, links, cum_links)
    magn_data[0, :n] = magnitude
    magn_data[0, n] = np.sum(magnitude)

    for t in range(1, runtime):
        apply_coin_step_inplace(state, n, links, cum_links, coins)
        state = state[shift_idx]
        magnitude = magn(state, n, links, cum_links)
        magn_data[t, :n] = magnitude
        magn_data[t, n] = np.sum(magnitude)

    mean_, std_, EE_, recur_avg_, recur1, recur2, recur3 = compute_stats_from_magnitudes(n, runtime, stabtime, magn_data)
    return mean_, std_, EE_, recur_avg_, recur1, recur2, recur3


def dtqw(n, N, links, cum_links, neighbour_list, g, run_count=RUN, runtime_val=runtime, stabtime_val=stabtime, max_workers=None, base_seed=42):
    shift_idx, coins = get_precomputed_ops(n, N, links, cum_links, neighbour_list)

    mean_runAvg = np.zeros(n)
    std_runAvg = np.zeros(n)
    EE_runAvg = np.zeros(n)
    recur_avg = np.zeros(n)
    recur1_all = []
    recur2_all = []
    recur3_all = []

    if max_workers is None:
        max_workers = cpu_count()

    seeds = [base_seed + i for i in range(run_count)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                simulate_one_run,
                seed,
                n,
                N,
                links,
                cum_links,
                neighbour_list,
                shift_idx,
                coins,
                runtime_val,
                stabtime_val,
                g,
            )
            for seed in seeds
        ]
        for f in as_completed(futures):
            mean_, std_, EE_, recur_avg_, recur1, recur2, recur3 = f.result()
            mean_runAvg += mean_
            std_runAvg += std_
            EE_runAvg += EE_
            recur_avg += recur_avg_
            if recur1 is not None and len(recur1) > 0:
                recur1_all = np.concatenate((recur1_all, recur1)) if len(recur1_all) else recur1
            if recur2 is not None and len(recur2) > 0:
                recur2_all = np.concatenate((recur2_all, recur2)) if len(recur2_all) else recur2
            if recur3 is not None and len(recur3) > 0:
                recur3_all = np.concatenate((recur3_all, recur3)) if len(recur3_all) else recur3

    denom = float(run_count)
    return mean_runAvg/denom, std_runAvg/denom, EE_runAvg/denom, recur_avg/denom, recur1_all, recur2_all, recur3_all


##############################################################################################  Run over various graphs
def parse_args():
    parser = argparse.ArgumentParser(description="DTQW on ER graph with multiprocessing")
    parser.add_argument("--n", type=int, default=n, help="number of nodes in graph")
    parser.add_argument("--m", type=int, nargs="+", default=[3], help="threshold for EE detection")
    parser.add_argument("--runs", type=int, default=RUN, help="number of independent runs")
    parser.add_argument("--seed", type=int, default=576711, help="random seed for graph generation")
    parser.add_argument("--base-seed", type=int, default=42, help="base seed for per-run RNG")
    parser.add_argument("--workers", type=int, default=None, help="max workers (processes); default=cpu count")
    parser.add_argument("--threads", type=int, default=None, help="threads for BLAS/OpenMP libs; default leaves as-is")
    return parser.parse_args()


def set_num_threads(threads):
    if threads is None:
        return
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)


if __name__ == "__main__":
    args = parse_args()
    set_num_threads(args.threads)

    n = args.n
    runtime = n*210
    stabtime = n*10
    RUN = args.runs
    m_vals = args.m

    for m in m_vals:
        print(f"Running for m = {m}")
        rnd = args.seed
        N, links, cum_links, neighbour_list = make_graph(n, rnd, m)
        mean_runAvg, std_runAvg, EE_runAvg, recur_avg, recur1_all, recur2_all, recur3_all = dtqw(
            n,
            N,
            links,
            cum_links,
            neighbour_list,
            m,
            run_count=RUN,
            runtime_val=runtime,
            stabtime_val=stabtime,
            max_workers=args.workers,
            base_seed=args.base_seed,
        )
        np.savetxt(f"mean_runAvg_n{n}_m{m}.csv", mean_runAvg, fmt='%.6f', delimiter=',', newline='\n')
        np.savetxt(f"std_runAvg_n{n}_m{m}.csv", std_runAvg, fmt='%.6f', delimiter=',', newline='\n')
        np.savetxt(f"EE_runAvg_n{n}_m{m}.csv", EE_runAvg, fmt='%.6f', delimiter=',', newline='\n')
        np.savetxt(f"recur_avg_k_n{n}_m{m}.csv", recur_avg, fmt='%.6f', delimiter=',', newline='\n')
        if recur1_all is not None and len(recur1_all) > 0:
            np.savetxt(f"recur1_n{n}_m{m}.csv", recur1_all, fmt='%.6f', delimiter=',', newline='\n')
        if recur2_all is not None and len(recur2_all) > 0:
            np.savetxt(f"recur2_n{n}_m{m}.csv", recur2_all, fmt='%.6f', delimiter=',', newline='\n')
        if recur3_all is not None and len(recur3_all) > 0:
            np.savetxt(f"recur3_n{n}_m{m}.csv", recur3_all, fmt='%.6f', delimiter=',', newline='\n')
        np.savetxt(f"links_n{n}_m{m}.csv", links, fmt='%.6f', delimiter=',', newline='\n')


