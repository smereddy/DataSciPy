import pandas as pd
import numpy as np

# Reading Data
test = pd.read_csv('DataSet/bigmart/Test.csv')
train = pd .read_csv('DataSet/bigmart/Train.csv')

# Concatenating Training and Test Data
data = pd.concat([train, test])
print(train.shape, test.shape, data.shape)

# Finding the count of missing value in the data by col
# Axis = 0 --> finding missing value in col
# Axis = 1 --> finding missng valur in row
count = data.isnull().sum(axis=0)
print(count)

# Checking statistic of the data
data_stat = data.describe()
print(data_stat)

unique = data.nunique()

print(unique)