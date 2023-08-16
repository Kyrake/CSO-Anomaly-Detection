# -*- coding: utf-8 -*-
"""data_exploration.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wzukcJO2PgVWzzEobN6KgHgi0iQwyHRL
"""

!pip install tensorflow==1.14.0

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
import random
from numpy import array
from keras import datasets
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from keras.callbacks import History
from keras.activations import relu
from keras.datasets import mnist
from keras.layers import Input, Dense, Conv2D, MaxPool2D, UpSampling2D, MaxPooling2D, Conv1D, MaxPooling1D, UpSampling1D
from keras.models import Model
from keras import backend as K
# make pandas only display two decimals
pd.options.display.float_format = '{:.2f}'.format

"""#Data 2017"""

#data16 = pd.read_excel(open('BWB_PI-Daten_wil2.xlsx', 'rb'), sheet_name='2016', header=1, skiprows=0)
#data16cso = pd.read_excel(open('CSO-Sim_wil.xlsx', 'rb'), sheet_name='2016', header=0)
data17 = pd.read_excel(open('BWB_PI-Daten_wil2.xlsx', 'rb'), sheet_name='2017', header=1, skiprows=0)
data17cso = pd.read_excel(open('CSO-Sim_wil.xlsx', 'rb'), sheet_name='2017', header=0)

# ~~ possibly concatenate the two datasets (2016 & 2017) here ~~

# in case we want to switch between sheets or concatenate them later
data = data17
datacso = data17cso

# set proper columns
data.columns = ['Time',
                  'Durchfluss RUH',
                  'Durchfluss STA',
                  'Durchfluss WAS',
                  'Wil Pegel ZK',
                  'Wil Pegel RB', # constant
                  'Wil Pegel ZK-RB', # constant
                  'Wil RM',
                  'Wil a RM',
                  'Wil m RM']

datacso.columns = ['Time',
                   'Seconds',
                   'Cso1',
                   'Cso2',
                   'Cso3',
                   'CsoSum']

# sanitize by setting "NaN" for all rows without real data
data["Wil Pegel ZK"] = pd.to_numeric(data["Wil Pegel ZK"], errors='coerce')
data["Wil Pegel RB"] = pd.to_numeric(data["Wil Pegel RB"], errors='coerce')
data["Wil Pegel ZK-RB"] = pd.to_numeric(data["Wil Pegel ZK-RB"], errors='coerce')
data["Wil RM"] = pd.to_numeric(data["Wil RM"], errors='coerce')
data["Wil a RM"] = pd.to_numeric(data["Wil a RM"], errors='coerce')
data["Wil m RM"] = pd.to_numeric(data["Wil m RM"], errors='coerce')

# compute additional columns

# differential rain meter data
data['Wil RM diff'] = data['Wil RM'].diff()
data['Wil a RM diff'] = data['Wil a RM'].diff()
data['Wil m RM diff'] = data['Wil m RM'].diff()

# differential pegel data (only ZK)
data['Wil Pegel ZK diff'] = data['Wil Pegel ZK'].diff()

# "combined" data of "interesting" columns + cso
cols = ['Durchfluss RUH',
         'Durchfluss STA',
         'Durchfluss WAS',
         'Wil Pegel ZK diff',
         'Wil Pegel ZK',
         'Wil RM diff']
combined = pd.concat([data[cols], datacso['CsoSum']], axis=1)

"""#Data2016"""

data16 = pd.read_excel(open('BWB_PI-Daten_wil2.xlsx', 'rb'), sheet_name='2016', header=1, skiprows=0)
data16cso = pd.read_excel(open('CSO-Sim_wil.xlsx', 'rb'), sheet_name='2016', header=0)
#data17 = pd.read_excel(open('BWB_PI-Daten_wil2.xlsx', 'rb'), sheet_name='2017', header=1, skiprows=0)
#data17cso = pd.read_excel(open('CSO-Sim_wil.xlsx', 'rb'), sheet_name='2017', header=0)

# ~~ possibly concatenate the two datasets (2016 & 2017) here ~~

# in case we want to switch between sheets or concatenate them later
data16 = data16
datacso16 = data16cso

# set proper columns
data16.columns = ['Time',
                  'Durchfluss RUH',
                  'Durchfluss STA',
                  'Durchfluss WAS',
                  'Wil Pegel ZK',
                  'Wil Pegel RB', # constant
                  'Wil Pegel ZK-RB', # constant
                  'Wil RM',
                  'Wil a RM',
                  'Wil m RM']

datacso16.columns = ['Time',
                   'Seconds',
                   'Cso1',
                   'Cso2',
                   'Cso3',
                   'CsoSum']

# sanitize by setting "NaN" for all rows without real data
data16["Wil Pegel ZK"] = pd.to_numeric(data16["Wil Pegel ZK"], errors='coerce')
data16["Wil Pegel RB"] = pd.to_numeric(data16["Wil Pegel RB"], errors='coerce')
data16["Wil Pegel ZK-RB"] = pd.to_numeric(data16["Wil Pegel ZK-RB"], errors='coerce')
data16["Wil RM"] = pd.to_numeric(data16["Wil RM"], errors='coerce')
data16["Wil a RM"] = pd.to_numeric(data16["Wil a RM"], errors='coerce')
data16["Wil m RM"] = pd.to_numeric(data16["Wil m RM"], errors='coerce')

# compute additional columns

# differential rain meter data
data16['Wil RM diff'] = data16['Wil RM'].diff()
data16['Wil a RM diff'] = data16['Wil a RM'].diff()
data16['Wil m RM diff'] = data16['Wil m RM'].diff()

# differential pegel data (only ZK)
data16['Wil Pegel ZK diff'] = data16['Wil Pegel ZK'].diff()

# "combined" data of "interesting" columns + cso
cols16 = ['Durchfluss RUH',
         'Durchfluss STA',
         'Durchfluss WAS',
         'Wil Pegel ZK diff',
         'Wil Pegel ZK',
         'Wil RM diff']
combined16 = pd.concat([data16[cols16], datacso16['CsoSum']], axis=1)

"""# Durchfluss Extraction 2017"""

data['DurchflussSum'] = data['Durchfluss RUH'] + data['Durchfluss STA'] + data['Durchfluss WAS']
durchfluss_sum =  data['DurchflussSum'].values.tolist()
durchfluss_sum = durchfluss_sum[1:]

print(np.amax(data['Durchfluss RUH'][1:]))
print(np.amax(data['Durchfluss STA'][1:]))
print(np.amax(data['Durchfluss WAS'][1:]))


timestamps_per_day = 288
data_time = data[["Time", 'DurchflussSum']].values.tolist()
data_time = data_time[1:]
#print(durchfluss_sum )
data_time = data_time[:288]
#print(data_time )
length = data["Time"].size
no_segment = int(np.floor(length / timestamps_per_day))
#print(no_segment)
list_time = []
list_durchfluss = []
for i in range(0, no_segment):
    one_day_time = data["Time"].values[i * 288:i * 288 + 288]
    one_day_time_strings = []
    for j in range(len(one_day_time)):
        one_day_time_j = pd.to_datetime(str(one_day_time[j]))
        one_day_time_strings.append(j)


        #print(one_day_time_j)

    one_day_durchfluss = durchfluss_sum[i * 288:i * 288 + 288]
    list_time.append(one_day_time_strings)
    list_durchfluss.append(one_day_durchfluss)

print(np.amax(list_durchfluss))
print(np.asarray(list_time).shape)
anomalies_list_time = list_time
#list_durchfluss = list_durchfluss[140:150]

"""# PEGEL ZK 2017"""

pegel_ZK =  data['Wil Pegel ZK'].values.tolist()

pegel_ZK = pegel_ZK[1:]

#print(np.amax(data['Durchfluss RUH'][1:]))
#print(np.amax(data['Durchfluss STA'][1:]))
#print(np.amax(data['Durchfluss WAS'][1:]))


timestamps_per_day = 288
data_time = data[["Time", 'Wil Pegel ZK']].values.tolist()
data_time = data_time[1:]

data_time = data_time[:288]
#print(data_time )
length = data["Time"].size
no_segment = int(np.floor(length / timestamps_per_day))
#print(no_segment)
list_time = []
list_pegel_ZK = []
for i in range(0, no_segment):
    one_day_time = data["Time"].values[i * 288:i * 288 + 288]
    one_day_time_strings = []
    for j in range(len(one_day_time)):
        one_day_time_j = pd.to_datetime(str(one_day_time[j]))
        one_day_time_strings.append(j)


        #print(one_day_time_j)

    one_day_pegel_ZK = pegel_ZK[i * 288:i * 288 + 288]
    list_time.append(one_day_time_strings)
    list_pegel_ZK.append(one_day_pegel_ZK)


#list_durchfluss = list_durchfluss[140:150]

#print(np.isnan(list_pegel_ZK))


list_pegel_ZK[155] = list_pegel_ZK[154]
list_pegel_ZK[101] = list_pegel_ZK[100]
#list_pegel_ZK.pop(155)
#list_pegel_ZK.pop(101)
print(list_pegel_ZK[155])
print(np.amax(list_pegel_ZK))



"""# CSO Extraction"""

datascolist =  datacso['CsoSum'].values.tolist()

datascolist = datascolist[1:]



timestamps_per_day = 288
data_time_cso = datacso[["Time", 'CsoSum']].values.tolist()
data_time_cso = data_time[1:]

data_time_cso = data_time_cso[:288]
#print(data_time_cso )
length = datacso["Time"].size
no_segment = int(np.floor(length / timestamps_per_day))
#print(no_segment)
list_time_cso = []
list_pegel_cso = []
for i in range(0, no_segment):
    one_day_time_cso = datacso["Time"].values[i * 288:i * 288 + 288]
    #print(one_day_time_cso )
    one_day_time_strings_cso = []
    for j in range(len(one_day_time_cso)):
        one_day_time_j_cso = pd.to_datetime(str(one_day_time_cso[j]))
        one_day_time_strings_cso.append(j)


        #print(one_day_time_j_cso)

    one_day_pegel_cso = datascolist[i * 288:i * 288 + 288]
    #print(one_day_time_j)
    list_time_cso.append(one_day_time_strings_cso)
    list_pegel_cso.append(one_day_pegel_cso)


#list_durchfluss = list_durchfluss[140:150]

#print(np.isnan(list_pegel_ZK))


print(list_pegel_cso)

anomaly_list = [i for i in range (len(list_pegel_cso)) if np.max(list_pegel_cso[i])>2]
print(anomaly_list)
print(len(anomaly_list))

"""#DataExtraction2016"""

data16['DurchflussSum'] = data16['Durchfluss RUH'] + data16['Durchfluss STA'] + data16['Durchfluss WAS']
print(np.amax(data16['Durchfluss RUH'][1:]))
print(np.amax(data16['Durchfluss STA'][1:]))
print(np.amax(data16['Durchfluss WAS'][1:]))
durchfluss_sum16 =  data16['DurchflussSum'].values.tolist()
durchfluss_sum16 = durchfluss_sum16[1:]
#durchfluss_sum16 = durchfluss_sum16[140:150]
timestamps_per_day = 288
data_time16 = data16[["Time", 'DurchflussSum']].values.tolist()
data_time16 = data_time16[1:]
#print(durchfluss_sum )
data_time16 = data_time16[:288]
#print(data_time )
length = data16["Time"].size
no_segment16 = int(np.floor(length / timestamps_per_day))
#print(no_segment)
list_time16 = []
list_durchfluss16 = []
for i in range(0, no_segment16):
    one_day_time16 = data16["Time"].values[i * 288:i * 288 + 288]
    one_day_time_strings16 = []
    for j in range(len(one_day_time16)):
        one_day_time_j16= pd.to_datetime(str(one_day_time16[j]))
        one_day_time_strings16.append(j)


        #print(one_day_time_j)

    one_day_durchfluss16 = durchfluss_sum16[i * 288:i * 288 + 288]
    list_time16.append(one_day_time_strings16)
    list_durchfluss16.append(one_day_durchfluss16)

print(np.amax(list_durchfluss16))
print(np.asarray(list_time16).shape)
#list_durchfluss16 = list_durchfluss16[140:150]

"""# Pegel ZK 2016"""

pegel_ZK16 =  data16['Wil Pegel ZK'].values.tolist()

pegel_ZK16 = pegel_ZK16[1:]

#print(np.amax(data['Durchfluss RUH'][1:]))
#print(np.amax(data['Durchfluss STA'][1:]))
#print(np.amax(data['Durchfluss WAS'][1:]))


timestamps_per_day = 288
data_time16 = data16[["Time", 'Wil Pegel ZK']].values.tolist()
data_time16 = data_time16[1:]

data_time16 = data_time16[:288]
#print(data_time )
length16 = data16["Time"].size
no_segment = int(np.floor(length / timestamps_per_day))
#print(no_segment)
list_timeZK16 = []
list_pegel_ZK16 = []
for i in range(0, no_segment):
    one_day_time16 = data16["Time"].values[i * 288:i * 288 + 288]
    one_day_time_strings16 = []
    for j in range(len(one_day_time16)):
        one_day_time_j16 = pd.to_datetime(str(one_day_time16[j]))
        one_day_time_strings16.append(j)


        #print(one_day_time_j)

    one_day_pegel_ZK16 = pegel_ZK16[i * 288:i * 288 + 288]
    list_timeZK16.append(one_day_time_strings16)
    list_pegel_ZK16.append(one_day_pegel_ZK16)
print(list_pegel_ZK16)
print(np.amax(list_pegel_ZK16))

#list_durchfluss = list_durchfluss[140:150]

"""#Plot2017Data"""

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, FuncFormatter)
list_time_plot = np.asarray(list_time)
list_pegel_ZK_plot = np.asarray(list_pegel_ZK)
row_list_time = np.arange(0,list_pegel_ZK_plot.flatten().shape[0]*5, 5)

major_ticks = np.arange(0, list_pegel_ZK_plot.flatten().shape[0]*5, 24*60)
fig, ax = plt.subplots()
#ax.set_xticks(major_ticks)
ax.grid(True, which='major', alpha = 0.8)
ax.xaxis.set_major_locator(MultipleLocator(24*60))
def format_func(value, ticknumber):
  return value/1440
ax.xaxis.set_major_formatter(FuncFormatter( format_func ))
plt.plot(row_list_time, list_pegel_ZK_plot.flatten())

plt.show()

plt.rcParams['figure.figsize'] = [300,5]
list_time_plot = np.asarray(list_time)
list_pegel_CSO_plot = np.asarray(list_pegel_cso)
row_list_time = np.arange(0,list_pegel_CSO_plot.flatten().shape[0]*5, 5)

major_ticks = np.arange(0, list_pegel_CSO_plot.flatten().shape[0]*5, 24*60)
fig, ax = plt.subplots()
#ax.set_xticks(major_ticks)
ax.grid(True, which='major', alpha = 0.8)
ax.xaxis.set_major_locator(MultipleLocator(24*60))
def format_func(value, ticknumber):
  return value/1440
ax.xaxis.set_major_formatter(FuncFormatter( format_func ))
plt.plot(row_list_time, list_pegel_CSO_plot.flatten())

"""#Druchfluss"""

no_segment =  180
fig, axs = plt.subplots(nrows= no_segment, ncols = 1, sharex= False, figsize=(10, 500))
ax1 = axs.flat

ax2 = ax1
ax3 = ax1

fig.tight_layout()
#fig.subplots_adjust(hspace = 0.2, wspace=1.4)

fig.tight_layout()
#fig.subplots_adjust(hspace = 0.2, wspace=1.4)
for i in range(no_segment):
        ax1[i].plot(list_time[i], list_durchfluss[i])
        ax1[i].set_ylim([np.min(list_durchfluss)-0.1, np.max(list_durchfluss)+0.1])
        ax2[i] =  ax1[i].twinx()
        ax2[i].plot(list_time[i], list_pegel_cso[i], color= 'tab:red')
        ax2[i].set_ylim([np.min(list_pegel_cso)-0.1, np.max(list_pegel_cso)+0.1])
        ax3[i] =  ax1[i].twinx()
        ax3[i].plot(list_time[i], list_pegel_ZK[i], color= 'tab:green')
        ax3[i].set_ylim([np.min(list_pegel_ZK)-0.1, np.max(list_pegel_ZK)+0.1])

        axs[i].set_title("DAY: %d" %i)

"""#ZK"""

import matplotlib.pyplot as plt
no_segment =  180
fig, axs = plt.subplots(nrows= no_segment, ncols = 1, sharex= False, figsize=(10, 500))
ax1 = axs.flat

ax2 = ax1


fig.tight_layout()
#fig.subplots_adjust(hspace = 0.2, wspace=1.4)
for i in range(no_segment):
        ax1[i].plot(list_time[i], list_pegel_ZK[i])
        ax1[i].set_ylim([np.min(list_pegel_ZK)-0.1, np.max(list_pegel_ZK)+0.1])
        ax2[i] =  ax1[i].twinx()
        ax2[i].plot(list_time[i], list_pegel_cso[i], color= 'tab:red')
        ax2[i].set_ylim([np.min(list_pegel_cso)-0.1, np.max(list_pegel_cso)+0.1])
        axs[i].set_title("DAY: %d" %i)

no_segment =  180
fig, axs = plt.subplots(nrows= no_segment, ncols = 1, sharex= False, figsize=(10, 500))
ax_flat = axs.flat

fig.tight_layout()
#fig.subplots_adjust(hspace = 0.2, wspace=1.4)
for i in range(no_segment):

        ax_flat[i].plot(list_time[i], list_pegel_cso[i])

        axs[i].set_title("DAY: %d" %i)

"""#Plot2016Data"""

no_segment =  180
fig, axs = plt.subplots(nrows= no_segment, ncols = 1, sharex= False, figsize=(10, 500))
ax_flat = axs.flat

fig.tight_layout()
#fig.subplots_adjust(hspace = 0.2, wspace=1.4)
for i in range(no_segment):
        ax_flat[i].plot(list_time16[i], list_durchfluss16[i])
        axs[i].set_title("DAY: %d" %i)

"""# Historical data

"""

#data.head()

# summary statistics
print("Summary Statistics:\n")
print(data.describe())

print("\n\nNaN counts:\n")
for col in data.columns:
    print("{:<16} {:d}".format(col, data[col].isna().sum()))

# Plot rainmeter data
columns = ['Wil RM', 'Wil a RM', 'Wil m RM']
axes = data[columns].plot(figsize=(16, 10), subplots=True) # marker='.', alpha=0.5, linestyle='None',
for ax in axes:
    ax.set_ylabel('Niederschlag (mm)')
plt.savefig('rainmeters.png')

# Plot rainmeter differential data
columns = ['Wil RM diff', 'Wil a RM diff', 'Wil m RM diff']
axes = data[columns].plot(figsize=(16, 10), subplots=True) # marker='.', alpha=0.5, linestyle='None',
for ax in axes:
    ax.set_ylabel('Niederschlag (mm)')
plt.savefig('rainmetersdiff.png')

# Plot durchfluss data
#columns = ['Durchfluss RUH', 'Durchfluss STA', 'Durchfluss WAS']
#axes = data[columns].plot(figsize=(16, 10), subplots=True)
#for ax in axes:
#    ax.set_ylabel('Durchfluss (l/s)')
#plt.savefig('durchfluesse.png')

# Plot Pegel data
columns = ['Wil Pegel ZK', 'Wil Pegel RB', 'Wil Pegel ZK-RB']
axes = data[columns].plot(figsize=(16, 10), subplots=True)
for ax in axes:
    ax.set_ylabel('Pegel (mNN)')
plt.savefig('pegels.png')

# boxplots for differential rain data
columns = ['Wil RM diff', 'Wil a RM diff', 'Wil m RM diff']
axes = data[columns].plot(figsize=(14, 4), kind='box', subplots=True)
plt.savefig('raindiffbox.png')

# distribution plots
#fig, ax = plt.subplots(figsize=(14,8))
#sns.distplot(data['Durchfluss RUH'], ax=ax)
#sns.distplot(np.nan_to_num(data['Wil Pegel ZK']), ax=ax[1])
#plt.savefig('durchflRUH_dist.png')

"""# Simulated (CSO) data"""

datacso.head()
# no NaN in this dataset

# Plot individual CSO data
columns = ['Cso1', 'Cso2', 'Cso3', 'CsoSum']
axes = datacso[columns].plot(figsize=(16, 10), subplots=True) # marker='.', alpha=0.5, linestyle='None',
for ax in axes:
    ax.set_ylabel('Water volume (m³/s)')
plt.savefig('csos.png')

"""# Combination of both datasets"""

# Rainmeter data + CSOs
temp = pd.concat([data[['Wil RM', 'Wil a RM', 'Wil m RM']],
           datacso['CsoSum']], axis=1)

# Plot individual CSO data
columns = ['Wil RM', 'Wil a RM', 'Wil m RM', 'CsoSum']
axes = temp[columns].plot(figsize=(16, 10), subplots=True) # marker='.', alpha=0.5, linestyle='None',
plt.savefig('raincso.png')

# Rainmeter differential + CSOs
temp = pd.concat([data[['Wil RM diff']],# 'Wil a RM diff', 'Wil m RM diff']], <- only RM
           datacso['CsoSum']], axis=1)

# Plot individual CSO data
columns = ['Wil RM diff', 'CsoSum'] #'Wil a RM diff', 'Wil m RM diff',
axes = temp[columns].plot(figsize=(16, 8), subplots=True) # marker='.', alpha=0.5, linestyle='None',
plt.savefig('raindiffcso.png')

temp = pd.concat([data[['Durchfluss RUH', 'Durchfluss STA', 'Durchfluss WAS']],
           datacso['CsoSum']], axis=1)

# Plot individual CSO data
columns = ['Durchfluss RUH', 'Durchfluss STA', 'Durchfluss WAS', 'CsoSum']
axes = temp[columns].plot(figsize=(16, 10), subplots=True) # marker='.', alpha=0.5, linestyle='None',
plt.savefig('durchflusscso.png', bbox_layout='tight')

temp = pd.concat([data['Wil Pegel ZK'],datacso['CsoSum']], axis=1)

# Plot individual CSO data
columns = ['Wil Pegel ZK', 'CsoSum']
axes = temp[columns].plot(figsize=(16, 8), subplots=True) # marker='.', alpha=0.5, linestyle='None',

temp = pd.concat([data['Wil Pegel ZK diff'],datacso['CsoSum']], axis=1)

# Plot individual CSO data
columns = ['Wil Pegel ZK diff', 'CsoSum']
axes = temp[columns].plot(figsize=(16, 8), subplots=True) # marker='.', alpha=0.5, linestyle='None',
plt.savefig('pegeldiffcso.png')

# Plot rainmeter data and rainmeter differentials (only interesting for Wil m RM of 2017 data)
columns = ['Wil m RM', 'Wil m RM diff']
axes = data[columns].plot(figsize=(16, 10), subplots=True)

temp = pd.concat([data, datacso['CsoSum']], axis=1)

#correlation matrix
corrmat = temp.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, square=True, vmin=-1.0, vmax=1.0);
plt.savefig('corrmat.png')

field1 = 'Wil Pegel ZK'
field2 = 'CsoSum'
temp = pd.concat([data[field1], datacso[field2]], axis=1)

#scatter plot totalbsmtsf/saleprice
temp.plot.scatter(x=field1, y=field2)
plt.savefig('pegelcsosum.png')

sns.pairplot(combined)#, size = 2.5)
#plt.show()
plt.savefig('pairplot.png')

#scatter plot totalbsmtsf/saleprice
data.plot.scatter(x='Durchfluss RUH', y='Wil Pegel ZK')

data.describe()

temp = pd.concat([data[['Durchfluss RUH', 'Wil RM diff', 'Wil Pegel ZK diff']],datacso['CsoSum']], axis=1)

# Plot individual CSO data
columns = ['Durchfluss RUH', 'Wil RM diff', 'Wil Pegel ZK diff', 'CsoSum']
axes = temp[columns].plot(figsize=(16, 12), subplots=True) # marker='.', alpha=0.5, linestyle='None',
plt.savefig('allcso.png')

temp = pd.concat([data[['Durchfluss RUH', 'Wil RM diff', 'Wil Pegel ZK diff']],datacso['CsoSum']], axis=1)

# Plot individual CSO data
columns = ['Durchfluss RUH', 'Wil RM diff', 'Wil Pegel ZK diff', 'CsoSum']
axes = temp[columns][20000:27000].plot(figsize=(16, 12), subplots=True) # marker='.', alpha=0.5, linestyle='None',
plt.savefig('allcso3.png')

"""## PCA + t-sne"""

# PCA
temp = data.copy()

pcacols = ['Durchfluss RUH',
           'Durchfluss STA',
           'Durchfluss WAS',
           'Wil Pegel ZK',
           'Wil RM',
           'Wil a RM']

# convert NaN's to 0
for c in pcacols:
    temp[c] = np.nan_to_num(temp[c])

pcadata = temp[pcacols].values

pca = PCA(n_components=3)
pca_result = pca.fit_transform(pcadata)

temp['pca-one'] = pca_result[:,0]
temp['pca-two'] = pca_result[:,1]
temp['pca-three'] = pca_result[:,2]

print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

ax = plt.figure(figsize=(16,10)).gca(projection='3d')
ax.scatter(
    xs=temp["pca-one"],
    ys=temp["pca-two"],
    zs=temp["pca-three"]#,
    #c=data.loc[rndperm,:]["y"],
    #cmap='tab10'
)
ax.set_xlabel('pca-one')
ax.set_ylabel('pca-two')
ax.set_zlabel('pca-three')
plt.show()

# t-SNE

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(pcadata)

temp['tsne-2d-one'] = tsne_results[:,0]
temp['tsne-2d-two'] = tsne_results[:,1]

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 10),
    data=temp,
    legend="full",
    alpha=0.3
)

plt.figure(figsize=(16,7))

ax1 = plt.subplot(1, 2, 1)
sns.scatterplot(
    x="pca-one", y="pca-two",
    palette=sns.color_palette("hls", 10),
    data=temp,
    legend="full",
    alpha=0.3,
    ax=ax1
)

ax2 = plt.subplot(1, 2, 2)
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    palette=sns.color_palette("hls", 10),
    data=temp,
    legend="full",
    alpha=0.3,
    ax=ax2
)

"""# Result: GARBO!

#Window
"""

window_size = (int)(6*60/5) #6 stunden in minuten schritten
list_pegel_ZK_np = np.asarray(list_pegel_ZK)
list_pegel_ZK_flat = list_pegel_ZK_np.flatten()
window = []
step_size = 3

for i in range (window_size):
  window.append(list_pegel_ZK_flat[i])

window_list = []
window_list.append(window)

count = 0
for i in range (window_size,len(list_pegel_ZK_flat)):
  window = window[1:]
  window.append(list_pegel_ZK_flat[i])
  count = count + 1
  if(count == step_size):
    window_list.append(window)
    count = 0


print(np.asarray(window_list).shape)
print(len(window_list)*0.65)

"""#LSTM 2017"""

import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense, RepeatVector,        TimeDistributed
from keras.layers import LSTM


x_train = window_list
x_train = np.asarray(x_train)/(np.nanmax(x_train))
print(x_train.shape)

n_features = 1
#print(x_train)


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],1))
x_train_uc = x_train
print(x_train)
timesteps = 72
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(timesteps,1), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(RepeatVector(timesteps))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
model.summary()


print(x_train.shape)

#model forf input to compute outpu

history = model.fit(x_train, x_train,
                epochs=25,
                batch_size=256,
                shuffle=False,
                validation_split=0.1)



##plot lossfunction
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# demonstrate reconstruction

print(x_train.shape)
yhat = model.predict(x_train)
print('---Predicted---')
print(np.round(yhat,3))
print('---Actual---')
print(np.round(x_train, 3))

print(yhat.shape)
train_mae_loss = np.mean(np.abs(np.round(yhat, 3) - np.round(x_train, 3)**2), axis=1)
print(train_mae_loss)
sns.distplot(train_mae_loss, bins = 50, kde = True);

threshold = 0.185

listi  = []

for i in range(len(x_train)):

  thisdict = {
    "loss": train_mae_loss[i],
    "threshold": threshold,
    "anomaly": train_mae_loss[i] > threshold,
    "value": x_train[i]
  }

  listi.append(thisdict)

row_list_time = np.arange(0,x_train.shape[0]*5, 5)
listi = np.asarray(listi)
print(len(listi))
print(row_list_time.shape)
row_loss = [listi[i]["loss"] for i in range(len(listi))]
row_threshold = [listi[i]["threshold"] for i in range(len(listi))]


plt.rcParams['figure.figsize'] = [10,10]
plt.plot(row_list_time, row_loss, label='loss')
plt.plot(row_list_time, row_threshold, label='threshold')
plt.xticks(rotation=25)
plt.legend();





list_of_anomalies = [i for i,x in enumerate(listi) if x["anomaly"][0] == True]
print(len(list_of_anomalies))
print(list_pegel_ZK_np.shape)
print(list_pegel_ZK_flat.shape)
print(x_train.shape)
# 288 werte pro tag
# da 6 std => 4 (nicht überlappende) windows pro tag
# da stepsize = 12
day_indeces = list(set([(int) ((i-1) / (4*24)) for i in list_of_anomalies]))
days_with_anomalies = [x for i,x in enumerate(list_pegel_ZK_np) if i in day_indeces]
days_with_anomalies_cso = [x for i,x in enumerate(list_pegel_cso) if i in day_indeces]
print((day_indeces))
print(len(days_with_anomalies_cso))

import matplotlib.pyplot as plt
no_segment =  len(days_with_anomalies)
fig, axs = plt.subplots(nrows= no_segment, ncols = 1, sharex= False, figsize=(150, 50))
ax1 = axs.flat

fig.tight_layout()
#fig.subplots_adjust(hspace = 0.2, wspace=1.4)
for i in range(no_segment):
        ax1[i].plot(anomalies_list_time[i], days_with_anomalies[i])
        ax1[i].set_ylim([np.min(days_with_anomalies)-0.1, np.max(days_with_anomalies)+0.1])
        ax2[i] =  ax1[i].twinx()
        ax2[i].plot(list_time[i], days_with_anomalies_cso[i], color= 'tab:red')
        ax2[i].set_ylim([np.min(days_with_anomalies_cso)-0.1, np.max(days_with_anomalies_cso)+0.1])
        axs[i].set_title("DAY: %d" %day_indeces[i])

"""# LSTM clean

"""

window_size = (int)(6*60/5) #6 stunden in minuten schritten
list_pegel_ZK_np = np.asarray(list_pegel_ZK)
list_pegel_ZK_np = np.delete(list_pegel_ZK_np, anomaly_list,0)
print(list_pegel_ZK_np.shape)
list_pegel_ZK_flat = list_pegel_ZK_np.flatten()/np.max(list_pegel_ZK_np)
window = []
step_size = 3

for i in range (window_size):
  window.append(list_pegel_ZK_flat[i])

window_list = []
window_list.append(window)

count = 0
for i in range (window_size,len(list_pegel_ZK_flat)):
  window = window[1:]
  window.append(list_pegel_ZK_flat[i])
  count = count + 1
  if(count == step_size):
    window_list.append(window)
    count = 0



print(np.asarray(window_list).shape)
print(window_list[0])

import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense, RepeatVector,        TimeDistributed
from keras.layers import LSTM


x_train = window_list
x_train = np.asarray(x_train)/(np.nanmax(x_train))
print(x_train.shape)

n_features = 1
#print(x_train)


x_train = x_train.reshape((x_train.shape[0], x_train.shape[1],1))
print(x_train)
timesteps = 72
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(timesteps,1), return_sequences=True))
model.add(LSTM(64, activation='relu', return_sequences=False))
model.add(RepeatVector(timesteps))
model.add(LSTM(64, activation='relu', return_sequences=True))
model.add(LSTM(128, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
model.summary()


print(x_train.shape)

#model forf input to compute outpu

history = model.fit(x_train, x_train,
                epochs=25,
                batch_size=256,
                shuffle=False,
                validation_split=0.1)



##plot lossfunction
print(history.history.keys())
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# demonstrate reconstruction
x_train = x_train_uc
yhat = model.predict(x_train)
print('---Predicted---')
print(np.round(yhat,3))
print('---Actual---')
print(np.round(x_train, 3))

print(yhat.shape)
train_mae_loss = np.mean(np.abs(np.round(yhat, 3) - np.round(x_train, 3)**2), axis=1)
print(train_mae_loss)
sns.distplot(train_mae_loss, bins = 50, kde = True);

threshold = 0.182

listi  = []

for i in range(len(x_train)):

  thisdict = {
    "loss": train_mae_loss[i],
    "threshold": threshold,
    "anomaly": train_mae_loss[i] > threshold,
    "value": x_train[i]
  }

  listi.append(thisdict)

row_list_time = np.arange(0,x_train.shape[0]*5, 5)
listi = np.asarray(listi)
print(len(listi))
print(row_list_time.shape)
row_loss = [listi[i]["loss"] for i in range(len(listi))]
row_threshold = [listi[i]["threshold"] for i in range(len(listi))]


plt.rcParams['figure.figsize'] = [10,10]
plt.plot(row_list_time, row_loss, label='loss')
plt.plot(row_list_time, row_threshold, label='threshold')
plt.xticks(rotation=25)
plt.legend();

list_of_anomalies = [i for i,x in enumerate(listi) if x["anomaly"][0] == True]
print(len(list_of_anomalies))
print(list_pegel_ZK_np.shape)
print(list_pegel_ZK_flat.shape)
print(x_train.shape)
# 288 werte pro tag
# da 6 std => 4 (nicht überlappende) windows pro tag
# da stepsize = 12
day_indeces = list(set([(int) ((i-1) / (4*24)) for i in list_of_anomalies]))
days_with_anomalies = [x for i,x in enumerate(list_pegel_ZK_np) if i in day_indeces]
days_with_anomalies_cso = [x for i,x in enumerate(list_pegel_cso) if i in day_indeces]
print((day_indeces))
print(len(days_with_anomalies_cso))

import matplotlib.pyplot as plt
no_segment =  len(days_with_anomalies)
fig, axs = plt.subplots(nrows= no_segment, ncols = 1, sharex= False, figsize=(150, 50))
ax1 = axs.flat

fig.tight_layout()
#fig.subplots_adjust(hspace = 0.2, wspace=1.4)
for i in range(no_segment):
        ax1[i].plot(anomalies_list_time[i], days_with_anomalies[i])
        ax1[i].set_ylim([np.min(days_with_anomalies)-0.1, np.max(days_with_anomalies)+0.1])
        ax2[i] =  ax1[i].twinx()
        ax2[i].plot(list_time[i], days_with_anomalies_cso[i], color= 'tab:red')
        ax2[i].set_ylim([np.min(days_with_anomalies_cso)-0.1, np.max(days_with_anomalies_cso)+0.1])
        axs[i].set_title("DAY: %d" %day_indeces[i])