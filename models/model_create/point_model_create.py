import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from google.colab import drive
drive.mount("/content/drive")

paifu = pd.read_csv("/content/drive/My Drive/mahjong/csv/haipai.csv", encoding="shift_jis")
paifu2 = pd.get_dummies(data=paifu, columns=["収支範囲"])
paifu2.head()

X = np.array(paifu2.drop(["局収支","収支範囲_-2","収支範囲_-1","収支範囲_1","収支範囲_2"], axis=1))
Y = np.array(paifu2[["収支範囲_-2","収支範囲_-1","収支範囲_1","収支範囲_2"]])
from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X_train.shape, Y_train.shape)
print(X_valid.shape, Y_valid.shape)

from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout
deep_model = keras.Sequential()
deep_model.add(Dense(38, activation="relu", input_shape=(38,)))
deep_model.add(Dropout(0.2))
deep_model.add(Dense(76, activation="relu"))
deep_model.add(Dropout(0.2))
deep_model.add(Dense(4, activation="softmax"))
deep_model.compile(optimizer="rmsprop",loss="categorical_crossentropy", metrics=["accuracy"])
deep_model.summary()

%%time
log = deep_model.fit(X_train, Y_train, epochs=1000, batch_size=64, verbose=True, \
                     callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=100, verbose=1)], \
                     validation_data=(X_valid, Y_valid))

Y_pred = deep_model.predict(X_valid)
Y_pred = (Y_pred * 100 + 0.5).astype(int)
df = pd.DataFrame(data=Y_pred, columns=["収支範囲_-2","収支範囲_-1","収支範囲_1","収支範囲_2"])
df

deep_model.save("/content/drive/My Drive/mahjong/model/point_model.h5")

df["収支範囲_1"].value_counts()