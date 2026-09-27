from check import *
check('fixqc',list(np.r_[np.arange(64.2,65.01,.1),np.arange(66.5,67.01,.05),np.arange(67.7,68.31,.05)]),(0,100,1420,460),.6,2,6)
check('qchead',list(np.arange(73.8,79.15,.1)),(1488,64,1836,285),1.2,4,5)
check('qcchestqr',list(np.r_[np.arange(72.3,73.15,.05),np.arange(73.6,73.86,.05),np.arange(74.93,76.21,.05)]),(1488,343,1836,565),1.2,4,5)
