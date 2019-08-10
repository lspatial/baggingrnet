from baggingrnet.model.bagging import  multBagging

inPath='/testpath/data/pm25_covs_train.csv'
gindex='gindex'
stratif='jd'

feasList = ['lat', 'lon', 'ele', 'prs', 'tem', 'rhu', 'win', 'pblh_re', 'pre_re', 'o3_re', 'aod', 'merra2_re', 'haod',
         'shaod', 'jd','lat2','lon2','latlon']
target='pm25_avg_log'
bagpath='/testpath/baggingrnet'
mbag=multBagging(bagpath)
mbag.getInputSample(inPath, feasList,stratif,gindex,target)

for i in range(80):
    name = str(i)
    nodes = [156,128,96,64,32,12]
    minibatch = 2560
    isresidual = True
    nepoch = 100
    sampling_fea = False
    noutput = 1
    mbag.addTask(name,noutput,sampling_fea, nepoch, nodes, minibatch, isresidual)

mbag.startMProcess(10)

