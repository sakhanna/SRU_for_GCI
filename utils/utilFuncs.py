# Import header files
import math
import torch
from torch import autograd
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import sys
import numpy as np
import pylab
from matplotlib import pyplot as plt 
import time
import sys
import csv

###########################################
# Python/numpy/pytorch environment config
###########################################
def env_config(GPUTrue, deviceName):
    
    global_seed = 2

    # Disable debug mode
    #torch.backends.cudnn.enabled=False
    torch.autograd.set_detect_anomaly(False)

    # Shrink very small values to zero in tensors for computational speedup
    torch.set_flush_denormal(True)

    # Set seed for random number generation (for reproducibility of results)
    torch.manual_seed(global_seed)
    torch.cuda.manual_seed(global_seed)
    np.random.seed(global_seed)

    # Set device as GPU if available, otherwise default to CPU
    if(GPUTrue):
        device = torch.device(deviceName if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    return device, global_seed



######################################
# Function for loading input data 
######################################
def loadTrainingData(inputDataFilePath, device):

    # Load and parse input data (create batch data)
    inpData = torch.load(inputDataFilePath)
    Xtrain = torch.zeros(inpData['TsData'].shape[1], inpData['TsData'].shape[0], requires_grad = False, device=device)
    Xtrain1 = inpData['TsData'].t()
    Xtrain.data[:,:] = Xtrain1.data[:,:]

    return Xtrain


    
#######################################################
# Function for reading ground truth network from file 
#######################################################
def loadTrueNetwork(inputFilePath, networkSize):

    with open(inputFilePath) as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')
        numrows = 0    
        for row in reader:
            numrows = numrows + 1

    network = np.zeros((numrows,2),dtype=np.int16)
    with open(inputFilePath) as tsvin:
        reader = csv.reader(tsvin, delimiter='\t')
        rowcounter = 0
        for row in reader:
            network[rowcounter][0] = int(row[0][1:])
            network[rowcounter][1] = int(row[1][1:])
            rowcounter = rowcounter + 1 

    Gtrue = np.zeros((networkSize,networkSize), dtype=np.int16)
    for row in range(0,len(network),1):
        Gtrue[network[row][1]-1][network[row][0]-1] = 1   
    
    return Gtrue




#############################################
# getCausalNodes
######################################
def getCausalNodes(model, threshold):
    n = model.n_inp_channels
    causalNodeMask = torch.zeros(n, 1, requires_grad = False, dtype=torch.int16)

    for col in range(n):
        #print(torch.norm(model.lin_xr2phi.weight.data[:,col],2))
        if(torch.norm(model.lin_xr2phi.weight.data[:,col], 2) > threshold):
            causalNodeMask.data[col] = 1
    return causalNodeMask


#######################################################################
# Calculates false positive negatives and true positives negatives
#####################################################################
def calcPerfMetrics(Gtrue, Gest):
    
    TP = 0 # True positive
    FP = 0 # False positive
    TN = 0 # True negative
    FN = 0 # False negative

    #n = Gest.shape[0]
    GTGE = (Gtrue * Gest)
    GestComplement = -1*(Gest-1)
    GtrueComplement = -1*(Gtrue-1)
    GTCGEC = (GtrueComplement * GestComplement)

    TP = np.sum(GTGE)
    FP = np.sum(Gest) - TP
    TN = np.sum(GTCGEC)
    FN = np.sum(GestComplement) - np.sum(GTCGEC)

    TPR = float(TP)/float(TP+FN)
    FPR = float(FP)/float(FP+TN)
    Recall = float(TP)/float(TP+FN)
    if(TP > 0 and FP > 0):
        Precision = float(TP)/float(TP+FP) 
    else:
        Precision = 0

    return TPR, FPR, Precision, Recall



####################################################
# Calculates area under ROC curve
# 
# (In) xin: numpy float array of false positive entries
# (In) yin: numpy float array of true positive entries
# (Out) auroc: calculated area under ROC curve
#
#  Notes: xin and yin should sorted and be of same dimension 
#         and contain bounded entries in (0,1)
####################################################
def calcAUROC(xin, yin, verbose):

    xin, yin = parallel_sort(xin, yin) 

    if(verbose > 0):
        for ii in range(len(xin)):
            print("%d\t %.6f \t %.6f" %(ii, xin[ii], yin[ii]))

     # Update input arrays to include extreme points (0,0) and (1,1) to the ROC plot
    xin = np.insert(xin,0,0)
    yin = np.insert(yin,0,0)
    xin = np.append(xin,1)
    yin = np.append(yin,1)

    n = len(xin)
    auroc = 0
    for ii in range(n-1):
        h = xin[ii+1]-xin[ii]
        b1 = yin[ii]
        b2 = yin[ii+1]
        trapezoid_area = 0.5*h*(b1 + b2)
        auroc = auroc + trapezoid_area
       
    return auroc, xin, yin


####################################################
# Calculates area under Precision-Recall curve
# 
# (In) xin: numpy float array of precision values
# (In) yin: numpy float array of recall values
# (Out) aupr: calculated area under precision-recall curve
#
#  Notes: xin and yin should sorted and be of same dimension 
#         and contain bounded entries in (0,1)
####################################################
def calcAUPR(xin, yin):

    ll = len(xin)

    # Update input arrays to include extreme points (0,1) and (1,0) to the precision-recall plot
    if(xin[0] > 0):
        xin = np.insert(xin,0,0)
        yin = np.insert(yin,0,1)
    if(xin[ll-1] < 1):    
        xin = np.append(xin,1)
        yin = np.append(yin,0)

    n = len(xin)
    aupr = 0
    for ii in range(n-1):
        h = xin[ii+1]-xin[ii]
        b1 = yin[ii]
        b2 = yin[ii+1]
        trapezoid_area = 0.5*h*(b1 + b2)
        aupr = aupr + trapezoid_area
            
    return aupr



###########################
# Count the number of tunable parameters in the model
##########################
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




##################################
# Calc metrics
##################################
def calcMetrics(jobLogFilename, model, dataset, verbose):

    ld = np.load(jobLogFilename)
    Gest = ld['Gest']
    Gref = ld['Gref']
    model_name = ld['model']
    dataset = ld['dataset']
    dsid = ld['dsid']
    nepochs = ld['nepochs']
    T = ld['T']
    F = ld['F']
    mu1 = ld['mu1']
    mu2 = ld['mu2']
    lr = ld['lr']

    # if esru2 model, then register esru2 specific parameters, namely mu3
    if(model_name == 'esru2' or model_name == 'esru3'):
        mu3 = ld['mu3']
    else:
        mu3 = 0

    n = Gest.shape[0]
    Gest1 = np.ones((n,n), dtype=np.int16)
    thresh = 0
    thresh_idx = (Gest <= thresh)
    Gest1.fill(1)
    Gest1[thresh_idx] = 0

    # remove self loops for gene causal network estimate
    if(dataset == 'gene'):
        for ii in range(n):
            Gest1[ii][ii] = 0

    #print(Gref)
    #print(Gest1)


    TPR, FPR, Precision, Recall = calcPerfMetrics(Gref, Gest1)
    if(verbose > 0):
        print("thresh = %1.4f, \t TPR = %1.4f, \t FPR = %1.4f, \t Precision = %.4f, \t Recall = %.4f" % (thresh, TPR, FPR, Precision, Recall))

    return model_name, dataset, dsid, T, F, nepochs, lr, mu1, mu2, mu3, TPR, FPR, Precision, Recall




##################################
# Calc metrics
##################################
def calcMetricsTCDF(jobLogFilename, model, dataset, threshold, verbose):

    ld = np.load(jobLogFilename)
    Gest = ld['Gest']
    Gref = ld['Gref']
    model_name = ld['model']
    dataset = ld['dataset']
    dsid = ld['dsid']
    nepochs = ld['nepochs']
    T = ld['T']
    F = ld['F']
    nepochs = ld['nepochs'] 
    kernel = ld['kernel_size']
    level = ld['levels']
    lr = ld['lr']
    dilation = ld['dilation_c']

    n = Gest.shape[0]
    Gest1 = np.ones((n,n), dtype=np.int16)
    thresh_idx = (Gest <= threshold)
    Gest1.fill(1)
    Gest1[thresh_idx] = 0

    # remove self loops for gene causal network estimate
    if(dataset == 'gene'):
        for ii in range(n):
            Gest1[ii][ii] = 0

    #print(Gref)
    #print(Gest1)

    TPR, FPR, Precision, Recall = calcPerfMetrics(Gref, Gest1)
    if(verbose > 0):
        print("thresh = %1.4f, \t TPR = %1.4f, \t FPR = %1.4f, \t Precision = %.4f, \t Recall = %.4f" % (thresh, TPR, FPR, Precision, Recall))

    return model_name, dataset, dsid, T, F, nepochs, lr, kernel, level, dilation, TPR, FPR, Precision, Recall




###################################################
# parallel sort in ascending order 
###################################################
def parallel_sort(xin, yin):

    n = len(xin)
    xin_sorted_idx = np.argsort(xin)
    yin_sorted_idx = np.argsort(yin)

    xout = xin[xin_sorted_idx]
    ysorted_by_x = yin[xin_sorted_idx]
    yout = yin

    #for ii in range(n):
    #    print("%d\t %.4f \t %.4f" %(ii, xout[ii], ysorted_by_x[ii]))

    # for fixed xin[.], further sort yin[...]    
    x_prev = xout[0]
    same_x_start_idx = 0
    yout=[]
    for ii in range(0, n, 1):
        x = xout[ii]
        if((x > x_prev) or (ii == n-1)):
            if(ii == n-1):
                same_x_stop_idx = n-1
            else:    
                same_x_stop_idx = ii-1
            
            if(same_x_start_idx == same_x_stop_idx):    
                y_arr_for_same_x = ysorted_by_x[same_x_start_idx]    
            else:
                y_arr_for_same_x = np.sort(ysorted_by_x[same_x_start_idx:same_x_stop_idx+1:1])
                #print("%d, %d, %.4f" %(same_x_start_idx, same_x_stop_idx, x_prev))
                #print(ysorted_by_x[same_x_start_idx:same_x_stop_idx+1:1])
                #print(y_arr_for_same_x)

            yout = np.append(yout, y_arr_for_same_x)

            #print("%d, %d, %.4f" %(same_x_start_idx, same_x_stop_idx, x_prev))
            same_x_start_idx = ii
            x_prev = xout[ii]

    return xout, yout




def getGeneTrainingData(dataset_id, device):

    if(dataset_id == 1):
        InputDataFilePath = "data/dream3/Dream3TensorData/Size100Ecoli1.pt"
        RefNetworkFilePath = "data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli1.tsv"
    elif(dataset_id == 2):
        InputDataFilePath = "data/dream3/Dream3TensorData/Size100Ecoli2.pt"
        RefNetworkFilePath = "data/dream3/TrueGeneNetworks/InSilicoSize100-Ecoli2.tsv"
    elif(dataset_id == 3):
        InputDataFilePath = "data/dream3/Dream3TensorData/Size100Yeast1.pt"
        RefNetworkFilePath = "data/dream3/TrueGeneNetworks/InSilicoSize100-Yeast1.tsv"
    elif(dataset_id == 4):
        InputDataFilePath = "data/dream3/Dream3TensorData/Size100Yeast2.pt"
        RefNetworkFilePath = "data/dream3/TrueGeneNetworks/InSilicoSize100-Yeast2.tsv"
    elif(dataset_id == 5):
        InputDataFilePath = "data/dream3/Dream3TensorData/Size100Yeast3.pt"
        RefNetworkFilePath = "data/dream3/TrueGeneNetworks/InSilicoSize100-Yeast3.tsv"
    else:
        print("Error while loading gene training data")    

    Xtrain = loadTrainingData(InputDataFilePath, device)
    n = Xtrain.shape[0]
    Gref = loadTrueNetwork(RefNetworkFilePath, n)   
    
    return Xtrain, Gref