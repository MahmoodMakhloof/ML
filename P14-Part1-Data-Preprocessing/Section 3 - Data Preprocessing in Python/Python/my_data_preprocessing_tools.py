# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 01:19:25 2021

@author: ma7mo
"""

# Data Preprocessing Tools

# importing libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[: , :-1].values
y = dataset.iloc[: ,  -1].values

# taking care of missing data
# replace it by mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values= np.nan, strategy='mean')
imputer = imputer.fit(X[ :, 1:3])
X[: , 1:3] = imputer.transform(X[ : , 1:3])
print('this is X after fill the nan by the mean\n',X,'\n','-'*50)

# encoding categorial data
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
# you can fit then transform or both in the same time
X[: ,0]=labelencoder_X.fit_transform(X[: , 0])
print('this is X after encode categorial data (country)\n',X,'\n','-'*50)
# contries encoded to numbers but the numbers that replaced to the countries will effect the data
# this is not valid so we need to make each category in single coulmns and put in it 0\1
from sklearn.preprocessing import OneHotEncoder
# here , we determine the column ([0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
print('this is X after hot encode categorial data (country)\n',X,'\n','-'*50)
# y encoding (yes/no) > (0/1)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print('this is y after encoding\n',y,'\n','-'*50)


# splitting dataset into the training and test
from sklearn.model_selection import train_test_split
# random_state = 42  (42 is recommended)
# if i wanna stop random >> random_state = 0 , shuffle = False , stratify = None
X_train , X_test ,y_train,y_test = train_test_split(X,y,test_size=0.2 , random_state =42)


# feature scaling
# Salary coulmn contain big numbers compared with age , So salary will affect more and the age will not affect at all
# make the data in the same scale (0~1)
# there are 2 ways (Standardisation & normalisation)
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test  = sc_X.transform(X_test)


