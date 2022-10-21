# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 15:36:38 2022

@author: Jungyu Lee, 301236221
Exercise#2: Commerce website predictions
"""
import pandas as pd

# a. get the data
ecom_exp_Jungyu = pd.read_csv('Ecom Expense.csv')

# b. initial exploration
ecom_exp_Jungyu.head(3)
ecom_exp_Jungyu.shape
ecom_exp_Jungyu.columns.values
ecom_exp_Jungyu.dtypes
for i in range(len(ecom_exp_Jungyu.columns.values)):
    if ecom_exp_Jungyu.isnull().any().tolist()[i] == False:
        print(ecom_exp_Jungyu.columns.values[i], 0)
        
# c. data transformation
ecom_exp_Jungyu = pd.concat([ecom_exp_Jungyu.drop('Gender', axis=1), pd.get_dummies(ecom_exp_Jungyu['Gender'])], axis=1)
ecom_exp_Jungyu = pd.concat([ecom_exp_Jungyu.drop('City Tier', axis=1), pd.get_dummies(ecom_exp_Jungyu['City Tier'])], axis=1)

ecom_exp_Jungyu.drop('Transaction ID', axis=1, inplace=True)

def normalize(x):
    return (x-x.min())/(x.max()-x.min())
    
normalized_Jungyu = normalize(ecom_exp_Jungyu)
normalized_Jungyu.head(2)

import matplotlib.pyplot as plt
normalized_Jungyu.hist(figsize=(9, 10))

pd.plotting.scatter_matrix(normalized_Jungyu[['Age', 'Monthly Income', 'Transaction Time', 'Total Spend']], alpha = 0.4, figsize=(13, 15))

# d. build a model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
feature_cols = ['Monthly Income', 'Transaction Time', 'Female', 'Male', 'Tier 1', 'Tier 2', 'Tier 3']
X = normalized_Jungyu[feature_cols]
Y = normalized_Jungyu['Total Spend']

x_train_Jungyu,x_test_Jungyu,y_train_Jungyu,y_test_Jungyu = train_test_split(X,Y, test_size = 0.35)

import numpy as np
np.random.seed(21)

lm = LinearRegression()
lm.fit(x_train_Jungyu, y_train_Jungyu)

lm.coef_
lm.score(x_train_Jungyu, y_train_Jungyu)

feature_cols2 = ['Monthly Income', 'Transaction Time', 'Female', 'Male', 'Tier 1', 'Tier 2', 'Tier 3', 'Record']
X2 = normalized_Jungyu[feature_cols2]
Y2 = normalized_Jungyu['Total Spend']

x_train_Jungyu2,x_test_Jungyu2,y_train_Jungyu2,y_test_Jungyu2 = train_test_split(X2,Y2, test_size = 0.35)
import numpy as np
np.random.seed(21)
lm = LinearRegression()
lm.fit(x_train_Jungyu2, y_train_Jungyu2)
lm.coef_
lm.score(x_train_Jungyu2, y_train_Jungyu2)





