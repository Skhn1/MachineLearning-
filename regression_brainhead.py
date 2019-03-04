# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 18:49:09 2019

@author: Shafaq Murtaza
"""

#%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(20.0,10.0)

#Reading Data
data=pd.read_csv('headbrain.csv')
print(data.shape)
data.head()
#Collecting X and Y
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values
#Mean of X and Y
mean_x=np.mean(X)
mean_y=np.mean(Y)

#Total number of values
m=len(X)

#Using the formula to calculate b1 and b2
numer=0
denom=0
for i in range(m):
    numer+=(X[i]-mean_x)*(Y[i]-mean_y)
    denom+=(X[i]-mean_x)**2
    
b1=numer/denom
b0=mean_y-(b1*mean_x)

#print coefficients
print(b1, b0)
#Plotting the values and Regression Line
max_x=np.max(X)+100
min_x=np.min(X)-100

#Calculating line values x and y
x=np.linspace(min_x, max_x, 1000)
y=b0+b1*x

#Plotting line
plt.plot(x,y,color='#58b970', label='Regression Line')

#Plotting scatter points
plt.scatter(X,Y,c='#ef5423', label='Scatter Plot')

plt.xlabel('Head Size in cm3')
plt.ylabel('Brain weight in grams')

plt.legend()
plt.show()
#Checking how good is our model by R-square method
#total sum of squares
ss_t=0
#total sum of residuals
ss_r=0
for i in range(m):
    y_pred=b0+b1*X[i]
    ss_t+=(Y[i]-mean_y)**2
    ss_r+=(Y[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print(r2)