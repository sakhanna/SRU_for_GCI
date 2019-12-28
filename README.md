# SRU_for_GCI
This repository contains the python code for our ICLR 2020 paper:[Economy Statistical Recurrent Units For Inferring Nonlinear Granger Causality](https://arxiv.org/abs/1911.09879).

## Dependencies
- Python 3.7
- CUDA 10.1
- PyTorch 1.0.1
- torchvision 0.2.1
- NumPy, SciPy

## Example usage: 
##### Model: SRU    Dataset: Lorenz-96, T=250, F=10, dataset id = 1
- python main.py --dataset lorenz --dsid 1 --model sru --n 10 --T 250 --F 10 --nepochs 2000 --mu1 0.021544 --mu2 0.031623 --mu3 0 --lr 0.001 --joblog crossval/logs/test.npz

##### Model: eSRU_2LF (Economy SRU with two-layer MLP as feedback's second stage)    Dataset: VAR, T=250, F=10, dataset id = 1
- python main.py --dataset var --dsid 1 --model eSRU_2LF --n 10 --T 500 --F 30 --nepochs 2000 --mu1 0.021544 --mu2 0.031623 --mu3 0.464159 --lr 0.001 --joblog crossval/logs/test.npz

##### Model: eSRU_2LF (Economy SRU with two-layer MLP as feedback's second stage)    Dataset: NetSim (BOLD signals), T=200, dataset id = 1
- python main.py --dataset netsim --dsid 1 --model eSRU_2LF --n 15 --T 200 --F 0 --nepochs 2000 --mu1 0.021544 --mu2 0.031623 --mu3 0.464159 --lr 0.001 --joblog crossval/logs/test.npz

##### Model: eSRU_1LF (Economy SRU with single layer MLP as feedback's second stage)    Dataset: Dream-3 (Yeast1), T=966 
- python main.py --dataset gene --dsid 1 --model eSRU_1LF --n 100 --T 966 --F 0 --nepochs 2000 --mu1 0.021544 --mu2 0.031623 --mu3 0.464159 --lr 0.001 --joblog crossval/logs/test.npz


## Input arguments

dataset :: string   :: lorenz/var/netsim/gene

dsid    :: int      :: dataset id (Range: 1-5)

model   :: string   :: sru (Standard SRU)
                       eSRU_1LF (Economy SRU with feedback's second stage implemented as single layer MLP)
                       eSRU_2LF (Economy SRU with feedback's second stage implemented as two layer MLP)

n       :: int 	    :: No. of timeseries/Nodes in the Granger causal graph 
                       (n = 10 for lorenz/var, n = 15 for netsim, n = 100 for gene)

T       :: int      :: Length of input timeseries 
                       (T = 250/500/1000 for lorenz, T =500/1000 for var, T = 200 for netsim, T = 966 for gene)

F       :: int      :: External forcing in lorenz model/ percentage sparsity in var model 
                       (F = 10/40 for lorenz, F = 30 for var, F = 0 for gene/netsim)

nepochs	:: int      ::  No. of training epochs

mu1     :: float    ::  Bias for ridge regularization of all unregularized weights in the model 

mu2     :: float    ::  Bias for block sparse regularization of input layer weights

mu3     :: float    ::  Bias for group sparse regularization of output feature layer weights in Economy SRU 

lr      :: float    ::  Learning rate for BPTT step in the proximal gradient descent algorithm


## Output 
- The adjacency matrix of the estimated Granger causal graph is printed as the output once the run is completed.
- A summary of model configuration and final results is stored in 'test.npz' located at: SRU_for_GCI_standalone\crossval\logs


## Acknowledgements
- The Dream-3 datasets for the gene interaction network inference experiments are taken from "http://dreamchallenges.org/project-list/dream3-2008/". 
- The BOLD-FMRI (NetSim) timeseries data used for the brain connectome inference experiments are taken from "https://www.fmrib.ox.ac.uk/datasets/netsim/index.html"  


