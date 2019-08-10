import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

inPath='/mnt/GAnaUS/PM25_jjj/data_sum/pm25_vars_v3_clear.csv'
orgDt=pd.read_csv(inPath,index_col='gindex')
mu, sigma = 0, 1
noise = np.random.normal(mu, sigma, [orgDt.shape[0],1])
orgDt['pm25_davgn']=orgDt['pm25_davg']+noise[:,0]
tinPath='/testpath/data/pm25_vars_v3_clear_random.csv'
orgDt.to_csv(tinPath,index=False)

trainIndex, testIndex =  train_test_split(range(orgDt.shape[0]), stratify=orgDt['jd'],test_size=0.11)

tinPath='/testpath/data/pm25_covs_train.csv'
orgDt.iloc[trainIndex].to_csv(tinPath,index=True,index_label='index')
tinPath='/testpath/data/pm25_covs_test.csv'
orgDt.iloc[testIndex].to_csv(tinPath,index=True,index_label='index')