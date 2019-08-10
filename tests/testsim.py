import numpy as np
from baggingrnet.data import data
from baggingrnet.model.bagging import  multBagging
import os
import shutil

def chkpath(tpath):
    if os.path.exists(tpath):
        shutil.rmtree(tpath)
    os.mkdir(tpath)

sim_train=data('sim_train')
sim_train['gindex']=np.array([i for i in range(sim_train.shape[0])])
sim_test=data('sim_test')
sim_test['gindex']=np.array([i for i in range(sim_test.shape[0])])

feasList = ['x'+str(i) for i in range(1,9)]
target='y'
bagpath='/testpath/baggingsim/res'
chkpath(bagpath)
mbag=multBagging(bagpath)
mbag.getInputSample(sim_train, feasList,None,'gindex',target)

name = str(0)
nodes = [32,16,8,4]
minibatch = 512
isresidual = True
nepoch = 200
sampling_fea = False
noutput = 1
islog=False
mbag.addTask(name,noutput,sampling_fea, nepoch, nodes, minibatch, isresidual,islog)

mbag.startMProcess(1)

from baggingrnet.model.baggingpre import  ensPrediction

target='pm25_avg_log'
prepath="/testpath/simprediction/res"
chkpath(prepath)
mbagpre=ensPrediction(bagpath,prepath)
mbagpre.getInputSample(sim_test, feasList,'gindex')
mbagpre.startMProcess(1)
mbagpre.aggPredict(isval=True,tfld='y')

