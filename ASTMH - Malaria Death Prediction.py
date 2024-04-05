#!/usr/bin/env python
# coding: utf-8

# # ASTMH: Time Series Visual Prediction of Malaria Death in Cote d’Ivoire using the HIMS Data

# In[1]:


# Importation of all the packages we need for this exercice

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.graphics.tsaplots as sgt 
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
sns.set()


# ## Explore the dataset

# In[2]:


# Importation of the time series data of Malaria death

df = pd.read_csv('D:/USAID/Poster_Presentation/Malaria_Death.csv')


# In[3]:


# How the data looking like

df.head(3)


# In[4]:


# Data type check

df.info()


# In[5]:


# The column 'Quarter' is in object format. We need to convert it in datetime format
df['Quarter'] = df['Quarter'].replace('q','Q')
df.head()


# In[6]:


df['date'] = pd.to_datetime(df['Quarter'])

df.head()


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


df['date'].dtype


# In[10]:


# Set the date column as an index of the dataset
df = df.set_index('date')


# In[11]:


df.head()


# In[12]:


df.tail()


# In[13]:


df_origin = df.copy()


# In[14]:


del df['Quarter']


# In[15]:


df.head()


# In[16]:


# Check the null values
df.isnull().sum()


# In[17]:


df.rename(columns={'Malaria death': 'Malaria_death'}, inplace=True)


# In[18]:


plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Malaria_death'], color='blue')
plt.title('Trend of Malaria Death in Cote dIvoire')
plt.xlabel('Date')
plt.ylabel('Number of Malaria Death')
plt.grid(True)
plt.show()


# ## Check the stationarity

# ### First test: Perform the rolling statistics

# In[19]:


df['rollMean'] = df['Malaria_death'].rolling(window=4).mean()
df['rollStd'] = df['Malaria_death'].rolling(window=4).std()


# In[20]:


plt.figure(figsize=(10,5))
sns.lineplot(data=df, x=df.index, y=df['Malaria_death'], color = 'blue', label='Malaria Death')
sns.lineplot(data=df, x=df.index, y=df['rollMean'], color = 'orange', label='Rolling Mean')
sns.lineplot(data=df, x=df.index, y=df['rollStd'], color = 'green', label='Rolling Std')
plt.legend()


# ### The data is not stationnary

# ### Second test: Perform the Augmented Dickey-Fuller (ADF) test

# In[21]:


# Import the ADF
from statsmodels.tsa.stattools import adfuller
from tabulate import tabulate


# In[22]:


result = adfuller(df['Malaria_death'])

table = [
    ['Test statistic', result[0]],
    ['p-value', result[1]],
    ['Test interpretation', 'The data is stationary' if result[1] < 0.05 else 'The data is not stationary']
]

# Tabulate the results
print(tabulate(table, headers=['Métrique', 'Valeur'], tablefmt='github'))


# In[23]:


# Perform seasonal breakdown
decomposition = seasonal_decompose(df['Malaria_death'], model='additive')

# Extract the components from the breakdown
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

# Display the breakdown components
plt.figure(figsize=(10, 5))

plt.subplot(411)
plt.plot(df['Malaria_death'], label='Original')
plt.legend(loc='best')

plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')

plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')

plt.subplot(414)
plt.plot(residual, label='Residues')
plt.legend(loc='best')

plt.tight_layout()
plt.show()


# # Making the data stationary

# ## Create a function to easily test the stationarity of the different methods

# In[24]:


def test_stationarity(dataFrame, var):
    dataFrame['rollMean'] = dataFrame[var].rolling(window=4).mean()
    dataFrame['rollstd'] = dataFrame[var].rolling(window=4).std()
    
    from statsmodels.tsa.stattools import adfuller
    adfTest = adfuller(dataFrame[var], autolag='AIC')
    stats = pd.Series(adfTest[0:4],index = ['The test statistic','p-value','#usedLags','#OBS'])
    print(stats)
    
    for key, values in adfTest[4].items():
        print('criticality',key,':',values)
        
    sns.lineplot(data=dataFrame, x=dataFrame.index, y=var, color = 'blue', label='var')    
    sns.lineplot(data=dataFrame, x=dataFrame.index, y='rollMean', color = 'orange', label='Rolling Mean')
    sns.lineplot(data=dataFrame, x=dataFrame.index, y='rollstd', color = 'green', label='Rolling Std')
    plt.legend()


# In[25]:


# Let's try if the function works well
test_stationarity(df,'Malaria_death')


# In[26]:


df.head(2)


# In[27]:


death_df = df[['Malaria_death']]
death_df.head(2)


# ## First method: Time shift transformation

# In[28]:


# Time Shift
death_df['shift'] = death_df['Malaria_death'].shift()
death_df['shiftdiff'] = death_df['Malaria_death'] - death_df['shift']
death_df.head(2)


# In[29]:


# Test the stationarity of the Time Shift Method
test_stationarity(death_df.dropna(),'shiftdiff')


# ### The shift transformation result is closed to a stationary data as the Test Statistic is less than the critical values and the p-value is less than 0.05. Let look the other data transformation results.

# ### Second method: Log transformation

# In[30]:


log_df = df[['Malaria_death']]
log_df['log'] = np.log(log_df['Malaria_death'])
log_df.head()


# In[31]:


test_stationarity(log_df,'log')


# ### The Log transformation does not make the data stationary.

# In[32]:


### The Square Root Transformation


# In[33]:


sqrt_df = df[['Malaria_death']]
sqrt_df['sqrt'] = np.sqrt(df['Malaria_death'])
sqrt_df.head()


# In[34]:


test_stationarity(sqrt_df,'sqrt')


# ### The Square Root transformation does not make the data stationary

# ### Let's try the Cube Root Transformation (cbrt)

# In[35]:


cbrt_df = df[['Malaria_death']]
cbrt_df['cbrt'] = np.cbrt(df['Malaria_death'])
cbrt_df.head()


# In[36]:


test_stationarity(cbrt_df,'cbrt')


# In[37]:


# Combinaison of Log and Square Root
log_df2 = log_df[['Malaria_death','log']]
log_df2['logsqrt'] = np.sqrt(log_df['log'])
log_df2.head()


# In[38]:


test_stationarity(log_df2,'logsqrt')


# In[39]:


# Combinaison of Log and Cube Root
log_df2 = log_df[['Malaria_death','log']]
log_df2['logcbrt'] = np.cbrt(log_df['log'])
log_df2.head()


# In[40]:


test_stationarity(log_df2,'logcbrt')


# In[41]:


death_df['shift'] = death_df['Malaria_death'].shift()
death_df['shiftdiff'] = death_df['Malaria_death'] - death_df['shift']
death_df['abshift'] = abs(death_df['shiftdiff'])
death_df.head()


# In[42]:


test_stationarity(death_df.dropna(),'abshift')


# ### The ShiftDift transformation shared the best result. Let's move forward with it.

# # Build the model

# In[43]:


death_df = df[['Malaria_death']].copy(deep=True)


# In[44]:


death_df['firstDiff'] = death_df['Malaria_death'].diff()
death_df['Diff4'] = death_df['Malaria_death'].diff(4)


# In[45]:


death_df['shift'] = death_df['Malaria_death'].shift()
death_df['shiftdiff'] = death_df['Malaria_death'] - death_df['shift']


# In[46]:


death_df.head()


# ## Buill ARIMA Model

# ### Figure Out Order For ARIMA Model

# In[47]:


# from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[48]:


plot_pacf(death_df['shiftdiff'].dropna(), lags=12);


# In[49]:


plot_acf(death_df['shiftdiff'].dropna(), lags=12);


# In[50]:


# We can install pmdarima to get the number of differencing

from pmdarima.arima.utils import ndiffs


# In[51]:


from statsmodels.tsa.stattools import adfuller


# In[52]:


ndiffs(death_df['Malaria_death'].dropna(), test='adf')


# In[53]:


# Based on the pacf and acf plots, and the ndiffs value, we can determine the different values of the ARIMA model.
# p= 3, or 4,or 10, or 12  d=1, q=4
# (3,1,4), or (4,1,4), or (10,1,4), (12,1,4)


# In[54]:


train = death_df[:round(len(death_df)*70/100)]
test = death_df[round(len(death_df)*70/100):]


# In[55]:


model = ARIMA(death_df['Malaria_death'],order = (12,1,4))
model_fit = model.fit()
prediction = model_fit.predict(start = test.index[0], end = test.index[-1])


# In[56]:


print(model_fit.summary())


# In[57]:


death_df['arimaPred'] = prediction
death_df.tail()


# In[58]:


death_df.dropna()
sns.lineplot(data=death_df, x=death_df.index, y='Malaria_death', color='blue', label='Malaria Death')
sns.lineplot(data=death_df, x=death_df.index, y='arimaPred', color='orange', label='ARIMA Prediction')
plt.legend()


# In[59]:


from sklearn.metrics import mean_squared_error


# In[60]:


np.sqrt(mean_squared_error(test['Malaria_death'], prediction))


# ### The Root Mean Square Error (RMSE) is low (less than 180). This means that the model is accurate.

# In[61]:


futureDate = pd.DataFrame(pd.date_range(start='2023-10-01', end='2024-08-30', freq='QS'), columns=['date'])
futureDate.set_index('date',inplace=True)
futureDate.head()


# In[62]:


import matplotlib.pyplot as plt
import numpy as np
import calendar


# In[63]:


model = ARIMA(death_df['Malaria_death'], order=(10,1,4))
results = model.fit()


# In[64]:


forecast_steps = 4 
forecast_index = pd.date_range(df.index[-1], periods=forecast_steps + 1, freq='Q')[1:] 
forecast = results.get_forecast(steps=forecast_steps, index=forecast_index) 


# In[65]:


forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()


# In[66]:


import matplotlib.pyplot as plt
import datetime
import matplotlib.units as munits
import matplotlib.dates as mdates
import matplotlib.dates as mdates
import matplotlib.cbook as cbook
import matplotlib.dates as mdates


# In[67]:


forecast = model_fit.predict(start=futureDate.index[0], end=futureDate.index[-1])
forecast


# In[ ]:





# In[68]:


plt.figure(figsize=(10, 6))
death_df.dropna()
sns.lineplot(data=death_df, x=death_df.index, y='Malaria_death', color='blue', label='Malaria Death')
sns.lineplot(data=death_df, x=death_df.index, y='arimaPred', color='orange',linestyle='dashed',linewidth=3, label='ARIMA Model test')
plt.axvline(datetime.datetime(2023, 10, 1), color='green',linestyle='dashed',linewidth=2,label='Original data trend limit')
sns.lineplot(data=forecast, x=forecast.index, y=forecast, color='red',linewidth=3, label='ARIMA Prediction')
# model_fit.predict(start=futureDate.index[0], end=futureDate.index[-1]).plot()
plt.legend()

