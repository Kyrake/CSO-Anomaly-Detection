import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, FuncFormatter)
import data_preparation as dp



def plot_cso(data):

    list_time_plot = np.asarray(data[4])
    list_pegel_ZK_plot = np.asarray(data[1])
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
    list_time_plot = np.asarray(data[1])
    list_pegel_CSO_plot = np.asarray(data[3])
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

def plot_cso_by_location(data, datacso):
    datacso.head()
    # no NaN in this dataset

    # Plot individual CSO data
    columns = ['Cso1', 'Cso2', 'Cso3', 'CsoSum']
    axes = datacso[columns].plot(figsize=(16, 10), subplots=True)  # marker='.', alpha=0.5, linestyle='None',
    for ax in axes:
        ax.set_ylabel('Water volume (m³/s)')
    plt.savefig('csos.png')

    """# Combination of both datasets"""

    # Rainmeter data + CSOs
    temp = pd.concat([data[['Wil RM', 'Wil a RM', 'Wil m RM']],
                      datacso['CsoSum']], axis=1)

    # Plot individual CSO data
    columns = ['Wil RM', 'Wil a RM', 'Wil m RM', 'CsoSum']
    axes = temp[columns].plot(figsize=(16, 10), subplots=True)  # marker='.', alpha=0.5, linestyle='None',
    plt.savefig('raincso.png')

    # Rainmeter differential + CSOs
    temp = pd.concat([data[['Wil RM diff']],  # 'Wil a RM diff', 'Wil m RM diff']], <- only RM
                      datacso['CsoSum']], axis=1)

    # Plot individual CSO data
    columns = ['Wil RM diff', 'CsoSum']  # 'Wil a RM diff', 'Wil m RM diff',
    axes = temp[columns].plot(figsize=(16, 8), subplots=True)  # marker='.', alpha=0.5, linestyle='None',
    plt.savefig('raindiffcso.png')

    temp = pd.concat([data[['Durchfluss RUH', 'Durchfluss STA', 'Durchfluss WAS']],
                      datacso['CsoSum']], axis=1)

    # Plot individual CSO data
    columns = ['Durchfluss RUH', 'Durchfluss STA', 'Durchfluss WAS', 'CsoSum']
    axes = temp[columns].plot(figsize=(16, 10), subplots=True)  # marker='.', alpha=0.5, linestyle='None',
    plt.savefig('durchflusscso.png', bbox_layout='tight')

    temp = pd.concat([data['Wil Pegel ZK'], datacso['CsoSum']], axis=1)

    # Plot individual CSO data
    columns = ['Wil Pegel ZK', 'CsoSum']
    axes = temp[columns].plot(figsize=(16, 8), subplots=True)  # marker='.', alpha=0.5, linestyle='None',

    temp = pd.concat([data['Wil Pegel ZK diff'], datacso['CsoSum']], axis=1)

    # Plot individual CSO data
    columns = ['Wil Pegel ZK diff', 'CsoSum']
    axes = temp[columns].plot(figsize=(16, 8), subplots=True)  # marker='.', alpha=0.5, linestyle='None',
    plt.savefig('pegeldiffcso.png')

    # Plot rainmeter data and rainmeter differentials (only interesting for Wil m RM of 2017 data)
    columns = ['Wil m RM', 'Wil m RM diff']
    axes = data[columns].plot(figsize=(16, 10), subplots=True)

    temp = pd.concat([data, datacso['CsoSum']], axis=1)

    # correlation matrix
    corrmat = temp.corr()
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, square=True, vmin=-1.0, vmax=1.0);
    plt.savefig('corrmat.png')

    field1 = 'Wil Pegel ZK'
    field2 = 'CsoSum'
    temp = pd.concat([data[field1], datacso[field2]], axis=1)

    # scatter plot totalbsmtsf/saleprice
    temp.plot.scatter(x=field1, y=field2)
    plt.savefig('pegelcsosum.png')

    sns.pairplot(data[0])  # , size = 2.5)
    # plt.show()
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


def plot_durchfluss(data):
    no_segment = 180
    fig, axs = plt.subplots(nrows=no_segment, ncols=1, sharex=False, figsize=(10, 500))
    ax1 = axs.flat

    ax2 = ax1
    ax3 = ax1

    fig.tight_layout()
    # fig.subplots_adjust(hspace = 0.2, wspace=1.4)

    fig.tight_layout()
    # fig.subplots_adjust(hspace = 0.2, wspace=1.4)
    for i in range(no_segment):
        ax1[i].plot(data[4][i], data[2][i])
        ax1[i].set_ylim([np.min(data[2]) - 0.1, np.max(data[2]) + 0.1])
        ax2[i] = ax1[i].twinx()
        ax2[i].plot(data[4][i], data[3][i], color='tab:red')
        ax2[i].set_ylim([np.min(data[3]) - 0.1, np.max(data[3]) + 0.1])
        ax3[i] = ax1[i].twinx()
        ax3[i].plot(data[4][i], data[1][i], color='tab:green')
        ax3[i].set_ylim([np.min(data[1]) - 0.1, np.max(data[1]) + 0.1])

        axs[i].set_title("DAY: %d" % i)

def plot_pegelZK(data):
    no_segment = 180
    fig, axs = plt.subplots(nrows=no_segment, ncols=1, sharex=False, figsize=(10, 500))
    ax1 = axs.flat

    ax2 = ax1

    fig.tight_layout()
    # fig.subplots_adjust(hspace = 0.2, wspace=1.4)
    for i in range(no_segment):
        ax1[i].plot(data[4][i], data[1][i])
        ax1[i].set_ylim([np.min(data[1]) - 0.1, np.max(data[1]) + 0.1])
        ax2[i] = ax1[i].twinx()
        ax2[i].plot(data[4][i], data[3][i], color='tab:red')
        ax2[i].set_ylim([np.min(data[3]) - 0.1, np.max(data[3]) + 0.1])
        axs[i].set_title("DAY: %d" % i)

    no_segment = 180
    fig, axs = plt.subplots(nrows=no_segment, ncols=1, sharex=False, figsize=(10, 500))
    ax_flat = axs.flat

    fig.tight_layout()
    # fig.subplots_adjust(hspace = 0.2, wspace=1.4)
    for i in range(no_segment):
        ax_flat[i].plot(data[1][i], data[3][i])

        axs[i].set_title("DAY: %d" % i)

def plot_rainmeter(data):
    columns = ['Wil RM', 'Wil a RM', 'Wil m RM']
    axes = data[columns].plot(figsize=(16, 10), subplots=True)  # marker='.', alpha=0.5, linestyle='None',
    for ax in axes:
        ax.set_ylabel('Niederschlag (mm)')
    plt.savefig('rainmeters.png')

    # Plot rainmeter differential data
    columns = ['Wil RM diff', 'Wil a RM diff', 'Wil m RM diff']
    axes = data[columns].plot(figsize=(16, 10), subplots=True)  # marker='.', alpha=0.5, linestyle='None',
    for ax in axes:
        ax.set_ylabel('Niederschlag (mm)')
    plt.savefig('rainmetersdiff.png')

    # Plot durchfluss data
    # columns = ['Durchfluss RUH', 'Durchfluss STA', 'Durchfluss WAS']
    # axes = data[columns].plot(figsize=(16, 10), subplots=True)
    # for ax in axes:
    #    ax.set_ylabel('Durchfluss (l/s)')
    # plt.savefig('durchfluesse.png')

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

