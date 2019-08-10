import numpy as np
from baggingrnet.data import data
from baggingrnet.model.bagging import  multBagging
import os
import shutil
import random as r

def chkpath(tpath):
    if os.path.exists(tpath):
        shutil.rmtree(tpath)
    os.mkdir(tpath)

pm25_train=data('pm2.5_train')
pm25_train['gindex']=np.array([i for i in range(pm25_train.shape[0])])
pm25_test=data('pm2.5_test')
pm25_test['gindex']=np.array([i for i in range(pm25_test.shape[0])])

feasList = ['lat', 'lon', 'ele', 'prs', 'tem', 'rhu', 'win', 'pblh_re', 'pre_re', 'o3_re', 'aod', 'merra2_re', 'haod',
         'shaod', 'jd','lat2','lon2','latlon']
target='pm25_avg_log'
bagpath='/testpath/baggingpm25_2/nores'
chkpath(bagpath)
mbag=multBagging(bagpath)
mbag.getInputSample(pm25_train, feasList,None,'gindex',target)

for i in range(1,81):
    name = str(i)
    nodes = [128 + r.randint(-5,5),128+ r.randint(-5,5),96,64,32,12]
    minibatch = 2560+r.randint(-5,5)
    isresidual = False
    nepoch = 120
    sampling_fea = False
    noutput = 1
    islog=True
    mbag.addTask(name,noutput,sampling_fea, nepoch, nodes, minibatch, isresidual,islog)

mbag.startMProcess(10)

from baggingrnet.model.baggingpre import  ensPrediction

prepath="/testpath/baggingpm25_2p/nores"
chkpath(prepath)
mbagpre=ensPrediction(bagpath,prepath)
mbagpre.getInputSample(pm25_test, feasList,'gindex')
mbagpre.startMProcess(10)
mbagpre.aggPredict(isval=True,tfld='pm25_davg')

