import pandas as pd
import numpy as np

print("libraries are loaded")

#step1 : loading data
train = pd.read_parquet('train.parquet')
test = pd.read_parquet('test.parquet')

print("Size of Train data:",train.shape)
print("Size of Test data:",test.shape)

print(train.head()) #first 5 rows of train

#step2 : health of data
print("missing values :", train.isnull().sum())
print("Data distribution : ", train['target'].value_counts())

#percentage
total = len(train)
anomaly = train['target'].value_counts()[1]
print(f"\nAnomaly percentage: {anomaly/total*100 :.2f}%")
