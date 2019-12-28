import math
import torch
import matplotlib
#import sys
import numpy as np
import pylab
from matplotlib import pyplot as plt 
#import time
from utilFuncs import loadTrueNetwork, getCausalNodes, calcPerfMetrics, calcAUROC, calcAUPR  

dataset = 'LORENZ'
#dataset = 'VAR'
#dataset = 'GENE'

if(dataset == 'LORENZ'):
    dataset_id = 1
    T = 1000
    F = 40.0
    model_name = 'sru'
    max_iter = 500
    n = 10
    thresh = 0.05
    muVec = np.arange(1.0, 11.0, 1.0)
    #muVec  = np.arange(18.0, 40.0, 1.0)
    TPRVec = np.zeros(len(muVec), dtype=np.float32) 
    FPRVec = np.zeros(len(muVec), dtype=np.float32) 
    RecallVec = np.zeros(len(muVec), dtype=np.float32)
    PrecisionVec = np.zeros(len(muVec), dtype=np.float32)
    Gest1  = np.zeros((n,n), dtype=np.int16)

    for muidx in range(len(muVec)):
        mu = muVec[muidx]
        LogPath = "logs/lorenz96/%s/%s%s_T%s_F%s_%s_niter%s_mu_%s.pt" % (model_name, dataset, dataset_id, T, F, model_name, max_iter, mu)
        savedTensors = torch.load(LogPath)
        Gref = savedTensors['Gref']
        Gest = savedTensors['Gest']
        Gest.requires_grad = False
        Gest1 = Gest.cpu().numpy()
        #print(Gest1)
        Gest1[Gest1 <= thresh] = 0
        Gest1[Gest1 > 0] = 1
        TPR, FPR, Precision, Recall = calcPerfMetrics(Gref, Gest1)
        print("mu = %.4f, \t thresh = %1.4f, \t TPR = %1.4f, \t FPR = %1.4f, \t Precision = %.4f, \t Recall = %.4f" % (mu, thresh, TPR, FPR, Precision, Recall))
        TPRVec[muidx] = TPR
        FPRVec[muidx] = FPR
        PrecisionVec[muidx] = Precision
        RecallVec[muidx] = Recall

    AUROC, FPRVec, TPRVec = calcAUROC(np.flip(FPRVec), np.flip(TPRVec))
    AUPR = calcAUPR(RecallVec, PrecisionVec)
    print("AUROC = %.4f, \t AUPR = %.4f" % (AUROC, AUPR))

    plt.figure(1)
    plt.title("ROC (TPR vs FPR)")
    plt.xlabel("False positive rate") 
    plt.ylabel("True positive rate")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(FPRVec, TPRVec)
    plt.show()

    plt.figure(2)
    plt.title("ROC (Precision vs Recall)")
    plt.xlabel("Recall") 
    plt.ylabel("Precision")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.plot(RecallVec, PrecisionVec)
    plt.show()


else:
    print("Dataset is not supported")






##########################
# old stuff
###########################
if 0:
    #LogPath = "logs/sru_niter1000_mu_2.0.pt"
    #LogPath = "logs/sru_mod_niter1000_mu_1.0.pt"
    #LogPath = "logs/lstm_niter200_mu_1.0.pt"
    #LogPath = "logs/lstm_niter200_mu_1.0.pt"
    #LogPath = "LORENZ_mlp_niter50000_mu_1.pt"
    InputDataFilePath = "Dream3TensorData/Size100Ecoli1.pt"
    RefNetworkFilePath = "Dream3TrueGeneNetworks/InSilicoSize100-Ecoli1.tsv"

    n = 100;
    Gref = loadTrueNetwork(RefNetworkFilePath, n)  
    savedTensors = torch.load(LogPath)
    Gest = savedTensors['Gest']
    print(Gest)
    Gest1 = torch.zeros(n,n, requires_grad = False, dtype=torch.int16)
    #thresholdVec = np.arange(0, 0.2, 0.001)
    thresholdVec = np.arange(0, 2, 0.05)
    TPRVec = np.zeros(len(thresholdVec), dtype=np.float32) 
    FPRVec = np.zeros(len(thresholdVec), dtype=np.float32) 
    RecallVec = np.zeros(len(thresholdVec), dtype=np.float32)
    PrecisionVec = np.zeros(len(thresholdVec), dtype=np.float32)

    #ignore self loops
    for ii in range(n):
        Gest.data[ii][ii] = 0



    for ii in range(len(thresholdVec)):
        thresh = thresholdVec[ii]
        for rr in range(n):
            for cc in range(n):
                Gest1.data[rr][cc] = Gest.data[rr][cc] > thresh
        TPR, FPR, Precision, Recall = calcPerfMetrics(Gref, Gest1)
        print("thresh = %1.4f, \t TPR = %1.4f, \t FPR = %1.4f, \t Precision = %.4f, \t Recall = %.4f" % (thresh, TPR, FPR, Precision, Recall))
        TPRVec[ii] = TPR
        FPRVec[ii] = FPR
        PrecisionVec[ii] = Precision
        RecallVec[ii] = Recall

  