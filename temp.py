# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as py
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('dummyDataSet.csv')
x=dataset.iloc[:,:-1].values
xx=dataset.iloc[:,:-1].values
xxx=dataset.iloc[:,:-1].values
y=dataset.iloc[:,3].values

import numpy as np
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='mean',fill_value='None')
imputer=imputer.fit(x[:,1:3])
x[:,1:3]=imputer.transform(x[:,1:3])
#for median 
import numpy as np
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='median',fill_value='None')
imputer=imputer.fit(xx[:,1:3])
xx[:,1:3]=imputer.transform(xx[:,1:3])
#for most_frequent
import numpy as np
from sklearn.impute import SimpleImputer
imputer=SimpleImputer(missing_values=np.nan,strategy='median',fill_value='None')
imputer=imputer.fit(xxx[:,1:3])
xxx[:,1:3]=imputer.transform(xxx[:,1:3])

# Now change city name to numbric from 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# transform the first column using LabelEncoder
labelencoder_x = LabelEncoder()
x[:,0] = labelencoder_x.fit_transform(x[:,0])

# transform the first column using OneHotEncoder
onehotencoder = OneHotEncoder()
x_onehot = onehotencoder.fit_transform(x[:,0].reshape(-1, 1)).toarray()

# replace the first column in x with the one-hot encoded values
x = np.concatenate((x_onehot, x[:,1:]), axis=1)

#traing and testing dataset (divided into two part 1 train and test )
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
  
#Stander and fit the data for better prediction
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_test=sc_x.fit_transform(x_test)
x_train=sc_x.fit_transform(x_train)