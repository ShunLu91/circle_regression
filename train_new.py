import numpy as np
import tensorflow as tf
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
import matplotlib.pyplot as plt
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization
import os
from keras.optimizers import SGD, Adam
from keras.models import model_from_json, Sequential


def draw_circle(img, row, col, rad):
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    valid = (
            (rr >= 0) &
            (rr < img.shape[0]) &
            (cc >= 0) &
            (cc < img.shape[1])
    )
    img[rr[valid], cc[valid]] = val[valid]


# just to test with
def clean_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)  # exclusive of high, in this case from 0 to 199
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    return (row, col, rad), img


def noisy_circle(size, radius, noise):
    img = np.zeros((size, size), dtype=np.float)

    # Circle
    row = np.random.randint(size)  # exclusive of high, in this case from 0 to 199
    col = np.random.randint(size)
    rad = np.random.randint(10, max(10, radius))
    draw_circle(img, row, col, rad)

    # Noise
    img += noise * np.random.rand(*img.shape)
    return (row, col, rad), img


def iou(params0, params1):
    row0, col0, rad0 = params0
    row1, col1, rad1 = params1

    shape0 = Point(row0, col0).buffer(rad0)
    shape1 = Point(row1, col1).buffer(rad1)

    return (
            shape0.intersection(shape1).area /
            shape0.union(shape1).area
    )


# 6 minutes when trained on Tesla K80 in Google Colab notebook
def train_model():
    img_size = 200
    rad = 50
    noise = 2
    numimages = 10000
    images = np.zeros((numimages, img_size, img_size), dtype=np.float)
    circle_locations = np.zeros((numimages, 1, 3),
                                dtype=np.float)  # num circles in image = 1, number of outputs req'd = 3 (r,c,rad)
    for i in range(numimages):
        loc, images[i] = noisy_circle(img_size, rad, noise)
        circle_locations[i] = np.array(loc).reshape(1, 3)
    # images.shape, circle_locations.shape

    X = images.reshape(images.shape[0], 200, 200, 1)
    Y = circle_locations.reshape(numimages, -1)  # /img_size #normalize to bring between 0 and 1

    split = int(0.8 * numimages)
    Xtrain = X[:split]
    Xtest = X[split:]
    Ytrain = Y[:split]
    Ytest = Y[split:]
    test_images = images[split:]
    test_locations = circle_locations[split:]
    test_images = test_images.reshape(test_images.shape[0], 200, 200, 1)

    cnn = Sequential()
    cnn.add(Conv2D(64, kernel_size=3, strides=(1, 1), data_format='channels_last', input_shape=(200, 200, 1),
                   padding='same', kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    cnn.add(Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    cnn.add(Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    cnn.add(Conv2D(512, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    cnn.add(Conv2D(512, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())
    cnn.add(Activation('relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    cnn.add(Flatten())
    cnn.add(Dense(1024, activation='relu'))
    cnn.add(Dense(512, activation='relu'))
    cnn.add(Dense(Y.shape[-1]))
    # cnn.compile(loss=tf.keras.losses.mean_squared_error,
    #             optimizer=tf.keras.optimizers.Adam(),
    #             metrics=['mse', 'mae', 'cosine'])
    cnn.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mse', 'mae', 'cosine'])
    cnn.fit(Xtrain, Ytrain,
            batch_size=64,
            epochs=100,
            verbose=2,
            validation_data=(Xtest, Ytest))

    return cnn


def find_circle(img, cnn):
    img = img.reshape(1, 200, 200, 1)
    pred = cnn.predict(img)
    pred = pred.reshape(-1)
    return pred


def main():
    if os.path.isfile('cnn_deeper.h5'):
        json_file = open('cnn_deeper.json', 'r')
        cnn_json = json_file.read()
        json_file.close()
        cnn = model_from_json(cnn_json)
        cnn.load_weights("cnn_deeper.h5")
    else:
        cnn = train_model()
        cnn_json = cnn.to_json()
        with open("cnn_deeper.json", "w") as json_file:
            json_file.write(cnn_json)
        cnn.save_weights("cnn_deeper.h5")
    results = []

    for _ in range(1000):
        params, img = noisy_circle(200, 50, 2)
        # noisy circles in 200*200 pixel matrix (ndarray), with radius from 10 to 50, and noise=2
        detected = find_circle(img, cnn)
        print(iou(params, detected))
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).sum())


main()