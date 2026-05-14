import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

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

#step4 - Prepare model

# Features list
FEATURES = ['X1', 'X2', 'logX3', 'logX4', 
            'month', 'dow', 'logX3_logX4_diff', 'X1_X2_ratio']

X = train[FEATURES].values
y = train['target'].astype(int).values

X_train, X_val, y_train, y_val = train_test_split(

    
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y

   
)

print("X_train shape:", X_train.shape)  
print("X_val shape  :", X_val.shape)    
print("y_train shape:", y_train.shape)  
print("y_val shape  :", y_val.shape)

from lightgbm import LGBMClassifier
spw = ((y_train==0).sum() / (y_train==1).sum())
print("Scale pos Weight :" , spw)

#Model
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    scale_pos_weight = spw,
    random_state=42,
    n_jobs=-1,
    verbose=-1

)

print("Model training...")
model.fit(X_train, y_train)
print("Model trained!")


