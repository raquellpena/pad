#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings(action='ignore')


# In[96]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style("darkgrid")
import statsmodels.api as sm
import scipy.stats as stats
from sklearn.metrics import r2_score
from pandas import read_csv
from numpy.linalg import inv
from numpy.linalg import det
from numpy import dot


# In[97]:


data = np.loadtxt('WineQT.csv', delimiter=',',skiprows=1)
data.shape


# In[98]:


x = data[:,1:2] # volatile acidity
y = data[:,8] # pH


# In[99]:


ax = sns.regplot(x=x, y=y, fit_reg=False, scatter_kws={'alpha':0.5})
ax.set(xlabel='Volatile Acidity',
       ylabel='pH');


# In[100]:


beta = dot(dot(inv(dot(x.T, x)),x.T),y) 
print('Estimated coefficient:', beta[0])


# In[101]:


beta = np.linalg.lstsq(x, y)[0]
print('Estimated coefficients:', beta[0])


# In[102]:


predictions = x * beta # making use of numpy's broadcast
predictions_withouth_intercept = predictions

ax = sns.regplot(x=x, y=y, fit_reg=False, scatter_kws={'alpha':0.5})
ax.set(xlabel='Volatile Acidity', 
       ylabel='pH',
       title='Linear Regression Relation bt. X & Y');
plt.plot(x, predictions) # overlay a line plot over a scatter plot 
plt.show()


# In[113]:


np.random.seed(0)
df = pd.DataFrame(data)
df.head()
pd.plotting.scatter_matrix(df, figsize=(30, 20))
plt.show()
plt.close()


# In[110]:





# In[ ]:




