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


data17 = pd.read_excel(open('BWB_PI-Daten_wil2.xlsx', 'rb'), sheet_name='2017', header=1, skiprows=0)
data17cso = pd.read_excel(open('CSO-Sim_wil.xlsx', 'rb'), sheet_name='2017', header=0)
data16 = pd.read_excel(open('BWB_PI-Daten_wil2.xlsx', 'rb'), sheet_name='2016', header=1, skiprows=0)
data16cso = pd.read_excel(open('CSO-Sim_wil.xlsx', 'rb'), sheet_name='2016', header=0)

def set_colums(data, datacso):
    data.columns = ['Time',
                    'Durchfluss RUH',
                    'Durchfluss STA',
                    'Durchfluss WAS',
                    'Wil Pegel ZK',
                    'Wil Pegel RB',  # constant
                    'Wil Pegel ZK-RB',  # constant
                    'Wil RM',
                    'Wil a RM',
                    'Wil m RM']

    datacso.columns = ['Time',
                       'Seconds',
                       'Cso1',
                       'Cso2',
                       'Cso3',
                       'CsoSum']

    return data, datacso

def clean_data(data, datacso):


    # sanitize by setting "NaN" for all rows without real data
    data["Wil Pegel ZK"] = pd.to_numeric(data["Wil Pegel ZK"], errors='coerce')
    data["Wil Pegel RB"] = pd.to_numeric(data["Wil Pegel RB"], errors='coerce')
    data["Wil Pegel ZK-RB"] = pd.to_numeric(data["Wil Pegel ZK-RB"], errors='coerce')
    data["Wil RM"] = pd.to_numeric(data["Wil RM"], errors='coerce')
    data["Wil a RM"] = pd.to_numeric(data["Wil a RM"], errors='coerce')
    data["Wil m RM"] = pd.to_numeric(data["Wil m RM"], errors='coerce')

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

    return combined



def durchfluss(data):
    data['DurchflussSum'] = data['Durchfluss RUH'] + data['Durchfluss STA'] + data['Durchfluss WAS']
    durchfluss_sum = data['DurchflussSum'].values.tolist()
    durchfluss_sum = durchfluss_sum[1:]

    timestamps_per_day = 288
    data_time = data[["Time", 'DurchflussSum']].values.tolist()
    data_time = data_time[1:]
    # print(durchfluss_sum )
    data_time = data_time[:288]
    # print(data_time )
    length = data["Time"].size
    no_segment = int(np.floor(length / timestamps_per_day))
    # print(no_segment)
    list_time = []
    list_durchfluss = []
    for i in range(0, no_segment):
        one_day_time = data["Time"].values[i * 288:i * 288 + 288]
        one_day_time_strings = []
        for j in range(len(one_day_time)):
            one_day_time_j = pd.to_datetime(str(one_day_time[j]))
            one_day_time_strings.append(j)

            # print(one_day_time_j)

        one_day_durchfluss = durchfluss_sum[i * 288:i * 288 + 288]
        list_time.append(one_day_time_strings)
        list_durchfluss.append(one_day_durchfluss)

    print(np.amax(list_durchfluss))
    print(np.asarray(list_time).shape)
    anomalies_list_time = list_time
    # list_durchfluss = list_durchfluss[140:150]

    return list_durchfluss

def pegel_zk(data):
    pegel_ZK = data['Wil Pegel ZK'].values.tolist()
    pegel_ZK = pegel_ZK[1:]
    timestamps_per_day = 288
    data_time = data[["Time", 'Wil Pegel ZK']].values.tolist()
    data_time = data_time[1:]

    data_time = data_time[:288]
    # print(data_time )
    length = data["Time"].size
    no_segment = int(np.floor(length / timestamps_per_day))
    # print(no_segment)
    list_time = []
    list_pegel_ZK = []
    for i in range(0, no_segment):
        one_day_time = data["Time"].values[i * 288:i * 288 + 288]
        one_day_time_strings = []
        for j in range(len(one_day_time)):
            one_day_time_j = pd.to_datetime(str(one_day_time[j]))
            one_day_time_strings.append(j)

            # print(one_day_time_j)

        one_day_pegel_ZK = pegel_ZK[i * 288:i * 288 + 288]
        list_time.append(one_day_time_strings)
        list_pegel_ZK.append(one_day_pegel_ZK)

    # list_durchfluss = list_durchfluss[140:150]

    # print(np.isnan(list_pegel_ZK))

    list_pegel_ZK[155] = list_pegel_ZK[154]
    list_pegel_ZK[101] = list_pegel_ZK[100]

    return list_pegel_ZK
def cso(datacso):
    datascolist = datacso['CsoSum'].values.tolist()

    datascolist = datascolist[1:]

    timestamps_per_day = 288
    data_time_cso = datacso[["Time", 'CsoSum']].values.tolist()
    data_time_cso = data_time_cso[1:]

    data_time_cso = data_time_cso[:288]
    # print(data_time_cso )
    length = datacso["Time"].size
    no_segment = int(np.floor(length / timestamps_per_day))
    # print(no_segment)
    list_time_cso = []
    list_pegel_cso = []
    for i in range(0, no_segment):
        one_day_time_cso = datacso["Time"].values[i * 288:i * 288 + 288]
        # print(one_day_time_cso )
        one_day_time_strings_cso = []
        for j in range(len(one_day_time_cso)):
            one_day_time_j_cso = pd.to_datetime(str(one_day_time_cso[j]))
            one_day_time_strings_cso.append(j)

            # print(one_day_time_j_cso)

        one_day_pegel_cso = datascolist[i * 288:i * 288 + 288]
        # print(one_day_time_j)
        list_time_cso.append(one_day_time_strings_cso)
        list_pegel_cso.append(one_day_pegel_cso)

    # list_durchfluss = list_durchfluss[140:150]

    # print(np.isnan(list_pegel_ZK))

    print(list_pegel_cso)

    anomaly_list = [i for i in range(len(list_pegel_cso)) if np.max(list_pegel_cso[i]) > 2]
    print(anomaly_list)
    return list_pegel_cso

def data_cso_2017(data17, data17cso):
    data, datacso = set_colums(data17, data17cso)
    combined17 = clean_data(data, datacso)
    durchflusss = durchfluss(data)
    pegel_ZK = pegel_zk(data)
    pegel_cso  = cso(datacso)
    return combined17, pegel_ZK, durchflusss, pegel_cso

def data_cso_2016(data16, data16cso):
    data, datacso = set_colums(data16, data16cso)
    combined16 = clean_data(data, datacso)
    durchflusss = durchfluss(data)
    pegel_ZK = pegel_zk(data)
    pegel_CSO = cso(datacso)
    return combined16, pegel_ZK, durchflusss, pegel_CSO