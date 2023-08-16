import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, RepeatVector, TimeDistributed
from keras.layers import LSTM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import data_preparation as dp
import matplotlib.pyplot as plt

data17 = dp.data_2017()
data16 = dp.data_2016()
def pca(data):
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

    temp['pca-one'] = pca_result[:, 0]
    temp['pca-two'] = pca_result[:, 1]
    temp['pca-three'] = pca_result[:, 2]

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

    ax = plt.figure(figsize=(16, 10)).gca(projection='3d')
    ax.scatter(
        xs=temp["pca-one"],
        ys=temp["pca-two"],
        zs=temp["pca-three"]  # ,
        # c=data.loc[rndperm,:]["y"],
        # cmap='tab10'
    )
    ax.set_xlabel('pca-one')
    ax.set_ylabel('pca-two')
    ax.set_zlabel('pca-three')
    plt.show()

    return pcadata

def tsne(data):
    temp = data.copy()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(pca(data))

    temp['tsne-2d-one'] = tsne_results[:, 0]
    temp['tsne-2d-two'] = tsne_results[:, 1]

    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        palette=sns.color_palette("hls", 10),
        data=temp,
        legend="full",
        alpha=0.3
    )

    plt.figure(figsize=(16, 7))

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

def windwoing(data):
    window_size = (int)(6 * 60 / 5)  # 6 stunden in minuten schritten
    list_pegel_ZK_np = np.asarray(data[2])
    list_pegel_ZK_flat = list_pegel_ZK_np.flatten()
    window = []
    step_size = 3

    for i in range(window_size):
        window.append(list_pegel_ZK_flat[i])

    window_list = []
    window_list.append(window)

    count = 0
    for i in range(window_size, len(list_pegel_ZK_flat)):
        window = window[1:]
        window.append(list_pegel_ZK_flat[i])
        count = count + 1
        if (count == step_size):
            window_list.append(window)
            count = 0
    return window_list

def lstm(data):
    x_train = windwoing(data)
    x_train = np.asarray(x_train) / (np.nanmax(x_train))
    print(x_train.shape)

    n_features = 1
    # print(x_train)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    print(x_train)
    timesteps = 72
    model = Sequential()
    model.add(LSTM(128, activation='relu', input_shape=(timesteps, 1), return_sequences=True))
    model.add(LSTM(64, activation='relu', return_sequences=False))
    model.add(RepeatVector(timesteps))
    model.add(LSTM(64, activation='relu', return_sequences=True))
    model.add(LSTM(128, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')
    model.summary()

    print(x_train.shape)

    # model forf input to compute outpu

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
    yhat = model.predict(x_train)
    return yhat

def prediction(data):
    x_train = windwoing(data)
    yhat = lstm(data)
    print('---Predicted---')
    print(np.round(yhat, 3))
    print('---Actual---')
    print(np.round(x_train, 3))

    print(yhat.shape)
    train_mae_loss = np.mean(np.abs(np.round(yhat, 3) - np.round(x_train, 3) ** 2), axis=1)
    print(train_mae_loss)
    sns.distplot(train_mae_loss, bins=50, kde=True);

    return train_mae_loss
def anomalies(data):
    threshold = 0.182
    train_mae_loss = prediction(data)
    x_train = windwoing(data)
    listi = []

    for i in range(len(x_train)):
        thisdict = {
            "loss": train_mae_loss[i],
            "threshold": threshold,
            "anomaly": train_mae_loss[i] > threshold,
            "value": x_train[i]
        }

        listi.append(thisdict)

    row_list_time = np.arange(0, x_train.shape[0] * 5, 5)
    listi = np.asarray(listi)
    print(len(listi))
    print(row_list_time.shape)
    row_loss = [listi[i]["loss"] for i in range(len(listi))]
    row_threshold = [listi[i]["threshold"] for i in range(len(listi))]

    plt.rcParams['figure.figsize'] = [10, 10]
    plt.plot(row_list_time, row_loss, label='loss')
    plt.plot(row_list_time, row_threshold, label='threshold')
    plt.xticks(rotation=25)
    plt.legend();

    list_of_anomalies = [i for i, x in enumerate(listi) if x["anomaly"][0] == True]
    print(len(list_of_anomalies))
    print(x_train.shape)
    # 288 werte pro tag
    # da 6 std => 4 (nicht überlappende) windows pro tag
    # da stepsize = 12
    day_indeces = list(set([(int)((i - 1) / (4 * 24)) for i in list_of_anomalies]))
    days_with_anomalies = [x for i, x in enumerate(data[1]) if i in day_indeces]
    days_with_anomalies_cso = [x for i, x in enumerate(data[3]) if i in day_indeces]
    print((day_indeces))
    print(len(days_with_anomalies_cso))

    no_segment = len(days_with_anomalies)
    fig, axs = plt.subplots(nrows=no_segment, ncols=1, sharex=False, figsize=(150, 50))
    ax1 = axs.flat
    ax2 = ax1

    fig.tight_layout()
    # fig.subplots_adjust(hspace = 0.2, wspace=1.4)
    for i in range(no_segment):
        ax1[i].plot(data[4][i], days_with_anomalies[i])
        ax1[i].set_ylim([np.min(days_with_anomalies) - 0.1, np.max(days_with_anomalies) + 0.1])
        ax2[i] = ax1[i].twinx()
        ax2[i].plot(data[4][i], days_with_anomalies_cso[i], color='tab:red')
        ax2[i].set_ylim([np.min(days_with_anomalies_cso) - 0.1, np.max(days_with_anomalies_cso) + 0.1])
        axs[i].set_title("DAY: %d" % day_indeces[i])