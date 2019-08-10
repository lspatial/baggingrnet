from baggingrnet.model.baggingpre import  ensPrediction


inPath='/testpath/data/pm25_covs_test.csv'
gindex='gindex'
feasList = ['lat', 'lon', 'ele', 'prs', 'tem', 'rhu', 'win', 'pblh_re', 'pre_re', 'o3_re', 'aod', 'merra2_re', 'haod',
         'shaod', 'jd','lat2','lon2','latlon']
target='pm25_avg_log'
bagpath='/testpath/baggingrnet'
prepath="/testpath/bagprediction1"
mbagpre=ensPrediction(bagpath,prepath)
mbagpre.getInputSample(inPath, feasList,gindex)
mbagpre.startMProcess(10)
mbagpre.aggPredict(isval=True,tfld='pm25_davg')


