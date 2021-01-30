import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import os
import cv2
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from google.colab import drive
drive.mount("/content/drive")

DIR = "/content/drive/My Drive/mahjong/images/"
CATEGORIES = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37"]
IMG_WIDTH = 50
IMG_HEIGHT = 65
train_data = []
X = []
Y = []

def create_train_data(dir, categories, data, X, Y):
    for category in categories:
        path = os.path.join(dir, category)
        for image_name in os.listdir(path):
            try:
                img = cv2.imread(os.path.join(path, image_name),)
                img_resize = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                train_data.append([img_resize, category])
            except Exception as e:
                pass
    for feature, label in data:
        X.append(feature)
        Y.append(label)

create_train_data(DIR, CATEGORIES, train_data, X, Y)
X = np.array(X)
Y = np.array(Y)
from tensorflow.keras.utils import to_categorical
Y2 = to_categorical(Y)
print(X.shape)
print(Y2.shape)

model = Sequential()
model.add(Conv2D(16, 3, padding="same", activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPooling2D())
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dense(38, activation="sigmoid"))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

from sklearn.model_selection import train_test_split
X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y2, test_size=0.2, random_state=0)

%%time
log = model.fit(X_train, Y_train, epochs=1000, batch_size=32, verbose=True, callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=100, verbose=1)], validation_data=(X_valid, Y_valid))

plt.plot(log.history['loss'], label='loss')
plt.plot(log.history['val_loss'], label='val_loss')
plt.legend(frameon=False) # 凡例の表示
plt.xlabel("epochs")
plt.ylabel("crossentropy")
plt.show()

model.save("/content/drive/My Drive/mahjong/image_model_50-65.h5")