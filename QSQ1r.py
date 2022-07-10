# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 22:18:05 2022

@author: Vincent
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy as scp


#Q1 (Prediction)

df1 = pd.read_csv('annual-motor-vehicle-inspection-passing-rate-of-motor-vehicles-on-first-inspection.csv')
df1_motorcycle = df1[df1['type']=='Motorcycles']
print(df1_motorcycle['passing_rate'].mean()) #passing rate is 90.74817109287878

df1_motorcycle_age = df1_motorcycle.groupby('age').sum()
''' to find new passing rate, use a function to edit the values'''
def passing(row):
    row['passing_rate'] = row['number_passed']/row['number_reported']
    return row['passing_rate']
df1_motorcycle_age['passing_rate']  = df1_motorcycle_age.apply(passing, axis = 1)
print(df1_motorcycle_age[['passing_rate']])

'''create 95% confidence interval for population mean weight'''
print(df1_motorcycle.loc[(df1_motorcycle['age'] == '4') & (df1_motorcycle['year'] == 2016)]) #7902 motorcycles for testing
df1_motorcycle_age5 = df1_motorcycle[df1_motorcycle['age'] == '5']
df1_sample = df1_motorcycle_age5[['number_reported', 'number_passed']] #ind variable is number_reported, dep variable is number_passed
sns.lmplot(x='number_reported',y='number_passed',data=df1_sample, fit_reg=True)
plt.title("Number reported vs number passed")
plt.show() #strong linear correlation observed, proceed to use Pearson for further testing
sns.residplot(x='number_reported',y='number_passed',data=df1_sample)
plt.title("Residual plot")
plt.show() #no structure present in residuals suggests that simple linear reg model is suitable
print(scp.stats.linregress(df1_sample)) #LinregressResult(slope=0.93485, intercept=5.61385, rvalue=0.99621, pvalue=6.09626e-12, stderr=0.025801, intercept_stderr=284.22433)
r,p = (scp.stats.pearsonr(df1_sample['number_reported'], df1_sample['number_passed'])) #(r = 0.996212, p = 6.09626e-12) #strong linear relation
z = np.arctanh(r) #z = 3.1336980873613087
se = 1/np.sqrt(df1_sample['number_reported'].size-3) #se = 0.3333333333333333

cint = z + np.array([-1,1]) * se * scp.stats.norm.ppf((1+0.95)/2)
cint1 = np.tanh(cint) #95% CFI is (2.480376759181290858e+00, 3.787019415541326595e+00)
passingrate_2018 = [7902*0.93485 + 5.61385]
CFI_2018 = np.append(passingrate_2018,cint)


