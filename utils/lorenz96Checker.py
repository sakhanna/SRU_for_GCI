import numpy as np
import torch
from utilFuncs import calcPerfMetrics, calcAUROC, calcAUPR  


# lorenz96 params
T = 1000
F = 40.0
model_name = 'lstm'
mu = 6.6  # F = 10, mu = 0.2| F = 40, mu = 4.0
n = 10
numDatasets = 5
max_iter = 500
verbose = 0

thresholdVec = np.arange(0, 1, 0.05)
#thresholdVec = np.arange(0, 0.1, 0.001)
TPRVec = np.zeros(len(thresholdVec), dtype=np.float32) 
FPRVec = np.zeros(len(thresholdVec), dtype=np.float32) 
RecallVec = np.zeros(len(thresholdVec), dtype=np.float32)
PrecisionVec = np.zeros(len(thresholdVec), dtype=np.float32)
Gest1 = np.ones((n,n), dtype=np.int16)
Gest2 = np.ones((n,n), dtype=np.int16)
Gref1 = np.zeros((n,n), dtype=np.int16)
AUROCList = np.zeros(numDatasets)
AUPRList = np.zeros(numDatasets)

for dsid in range(numDatasets):
    filename = "../logs/lorenz96/LORENZ%s_T%s_F%s_%s_niter%s_mu_%s.pt" % (dsid+1, T, F, model_name, max_iter, mu)
    savedTensors = torch.load(filename)
    Gest = savedTensors['Gest']
    Gref = savedTensors['Gref']
    Gest.requires_grad = False
    Gest1 = Gest.cpu().numpy()
    Gest2.fill(1)

    for ii in range(len(thresholdVec)):
        thresh = thresholdVec[ii]
        thresh_idx = (Gest1 < thresh)
        Gest2[thresh_idx] = 0
        TPR, FPR, Precision, Recall = calcPerfMetrics(Gref, Gest2)
        if(verbose > 0):
            print("thresh = %1.4f, \t TPR = %1.4f, \t FPR = %1.4f, \t Precision = %.4f, \t Recall = %.4f" % (thresh, TPR, FPR, Precision, Recall))
        TPRVec[ii] = TPR
        FPRVec[ii] = FPR
        PrecisionVec[ii] = Precision
        RecallVec[ii] = Recall

    AUROCList[dsid] = calcAUROC(np.flip(FPRVec), np.flip(TPRVec))
    AUPRList[dsid] = calcAUPR(RecallVec, PrecisionVec)
    print("%s_LORENZ%d_T%s_F%s: AUROC = %.4f, \t AUPR = %.4f" % (model_name, dsid, T, F, AUROCList[dsid], AUPRList[dsid]))


print("Mean AUROC = %.4f, \t Mean AUPR = %.4f" % (AUROCList.mean(), AUPRList.mean()))


