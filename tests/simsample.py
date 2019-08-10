import numpy as np
from baggingrnet.data.simulatedata import simData
from sklearn.model_selection import train_test_split

nsample=12000
simdata=simData(nsample)
simdata['gindex']=np.array([i for i in range(nsample)])

trainIndex, testIndex =  train_test_split(range(nsample),test_size=0.2)

simdataTrain=simdata.iloc[trainIndex]
simdataTest=simdata.iloc[testIndex]

tfl="/dpLearnPrj/package_dev/baggingrnet/baggingrnet/data/sim_train.csv"
simdataTrain.to_csv(tfl,index=True,index_label='index')
tfl="/dpLearnPrj/package_dev/baggingrnet/baggingrnet/data/sim_test.csv"
simdataTest.to_csv(tfl,index=True,index_label='index')