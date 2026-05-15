import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, confusion_matrix

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

y_pred = model.predict(X_val)

acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred, zero_division=0)
f1 = f1_score(y_val, y_pred, zero_division=0)
rec = recall_score(y_val, y_pred, zero_division=0)

print("=== Model Evaluation ===")
print(f"Accuracy  : {acc*100:.2f}%")
print(f"Precision : {prec*100:.2f}%")
print(f"Recall    : {rec*100:.2f}%")
print(f"F1 Score  : {f1*100:.2f}%")

# Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
print("\nConfusion Matrix:")
print(cm)

#Threshold tuning
probs = model.predict_proba(X_val)[:,1]

best_t, best_f1 = 0.5, 0
for t in np.arange(0.01, 0.99, 0.01):
    preds = (probs >= t).astype(int)
    f1 = f1_score(y_val, preds, zero_division=0)
    if f1 > best_f1:
        best_f1, best_t = f1, t

print(f"Best Threshold : {best_t:.2f}")
print(f"Best F1 Score  : {best_f1*100:.2f}%")

# Test data prediction
X_test_final = test[FEATURES].values
test_probs = model.predict_proba(X_test_final)[:,1]
test_preds = (test_probs >= best_t).astype(int)

print("Test predictions:")
print("Normal (0)  :", (test_preds == 0).sum())
print("Anomaly (1) :", (test_preds == 1).sum())

# Submission file
submission = pd.DataFrame({
    'ID'    : test['ID'].values,
    'target': test_preds.astype(str)
})

submission.to_csv('submission_learning.csv', index=False)
print("\nSubmission saved!")
print(submission.head())