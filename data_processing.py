import numpy as np
import keras
from keras import Sequential
from keras.layers import Dense, RepeatVector,        TimeDistributed
from keras.layers import LSTM


def pca():
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

def tsne():
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(pcadata)

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


def windwoing():
    window_size = (int)(6 * 60 / 5)  # 6 stunden in minuten schritten
    list_pegel_ZK_np = np.asarray(list_pegel_ZK)
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

def lstm():
    x_train = window_list
    x_train = np.asarray(x_train) / (np.nanmax(x_train))
    print(x_train.shape)

    n_features = 1
    # print(x_train)

    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_train_uc = x_train
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