import pandas as pd
import numpy as np

train = pd.read_parquet('train.parquet')
test = pd.read_parquet('test.parquet')

#step3 - feature engineering

train['logX3']= np.log(train['X3'].clip(1e-9))
train['logX4']= np.log(train['X4'].clip(1e-9))
test['logX3'] = np.log(test['X3'].clip(1e-9))
test['logX4'] = np.log(test['X4'].clip(1e-9))

train['month']= train['Date'].dt.month
train['dow']= train['Date'].dt.dayofweek
test['month']=test['Date'].dt.month
test['dow']= test['Date'].dt.dayofweek

train['logX3_logX4_diff'] = train['logX3'] - train['logX4']
train['X1_X2_ratio']      = train['X1'] / (train['X2'] + 1e-9)
test['logX3_logX4_diff']  = test['logX3'] - test['logX4']
test['X1_X2_ratio']       = test['X1'] / (test['X2'] + 1e-9)

print("Features before:", ['X1','X2','X3','X4','X5'])
print("Features after :", ['X1','X2','logX3','logX4','month','dow','logX3_logX4_diff','X1_X2_ratio'])
print("\nNew features added successfully!")

