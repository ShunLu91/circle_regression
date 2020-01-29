import re
import numpy as np
import argparse as arg
import tensorflow as tf
import os, glob, datetime
import keras.layers as KL
import keras.backend as K
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, LearningRateScheduler
from shapely.geometry.point import Point
from skimage.draw import circle_perimeter_aa
from main import noisy_circle, draw_circle, iou

K.clear_session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
np.random.seed(2020)


def data_generator(samples, size, radius, noise):
    params = []
    images = []
    for _ in range(samples):
        para, img = noisy_circle(size, radius, noise)
        params.append(np.asarray(para))
        images.append(img)
    images = np.array(images, dtype=np.float)
    images = images.reshape((-1, images.shape[1], images.shape[2], 1))
    # params = np.array(params, dtype=np.float64) / size
    params = np.array(params, dtype=np.float64)
    return images, params


size = 200
x_train, y_train = data_generator(10000, size, 50, 2)
print('x_train:', x_train.shape, 'y_train:', y_train.shape)

# parser = argparse.ArgumentParser()
# args = parser.parse_args()
save_dir = os.path.join('models')

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


def model_graph(size, filters, image_channels=1):
    inpt = KL.Input(shape=(size, size, image_channels), name='input_images')

    x = KL.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', name='conv1')(inpt)
    x = KL.Activation('relu', name='relu1')(x)
    x = KL.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', name='conv2')(x)
    x = KL.Activation('relu', name='relu2')(x)
    #     x = KL.BatchNormalization(axis=-1, momentum=0.0,epsilon=0.0001, name = 'bn1')(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = KL.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='conv3')(x)
    x = KL.Activation('relu', name='relu3')(x)
    x = KL.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same', name='conv4')(x)
    x = KL.Activation('relu', name='relu4')(x)
    #     x = KL.BatchNormalization(axis=-1, momentum=0.0,epsilon=0.0001, name = 'bn2')(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    #     x = KL.Conv2D(filters=64, kernel_size=(3,3), strides=1, padding='same',name = 'conv5')(x)
    #     x = KL.Activation('relu',name = 'relu5')(x)
    #     x = KL.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = KL.Flatten()(x)
    x = KL.Dense(1024, activation='relu')(x)
    x = KL.Dropout(0.5)(x)
    x = KL.Dense(3)(x)
    model = Model(inputs=inpt, outputs=x)

    return model


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.hdf5'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).hdf5.*", file_)
            # print(result[0])
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


def lr_schedule(epoch):
    initial_lr = 0.001
    if epoch <= 50:
        lr = initial_lr
    elif epoch <= 100:
        lr = initial_lr / 10
    elif epoch <= 200:
        lr = initial_lr / 20
    elif epoch <= 300:
        lr = initial_lr / 30
    elif epoch <= 350:
        lr = initial_lr / 50
    elif epoch <= 400:
        lr = initial_lr / 80
    log('current learning rate is %2.8f' % lr)
    return lr


model = model_graph(size, filters=32)
# model.summary()

initial_epoch = findLastCheckpoint(save_dir=save_dir)
if initial_epoch > 0:
    print('resuming by loading epoch %03d' % initial_epoch)
    model = load_model(os.path.join(save_dir, 'model_%03d.hdf5' % initial_epoch), compile=False)

# compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
checkpointer = ModelCheckpoint(os.path.join(save_dir, 'model_{epoch:03d}.hdf5'), verbose=1,
                               save_weights_only=False, period=10)
csv_logger = CSVLogger(os.path.join(save_dir, 'log.csv'), append=True, separator=',')
lr_scheduler = LearningRateScheduler(lr_schedule)
history = model.fit(x_train, y_train, initial_epoch=initial_epoch, shuffle=True, validation_split=0.3, batch_size=4,
                    epochs=100, verbose=1, callbacks=[checkpointer, csv_logger, lr_scheduler])

# model = load_model(os.path.join('models', 'model_010.hdf5'), compile=False)
results = []
for _ in range(1000):
    params, img = noisy_circle(200, 50, 2)
    # detected = find_circle(img)
    # print(img)
    # img = img.reshape((-1, img.shape[0], img.shape[1], 1))
    # x_ts = []
    # y_ts = []
    # x_ts.append(np.asarray(params))
    # y_ts.append(img)
    # y_ts = np.array(y_ts, dtype=np.float)
    # y_ts = y_ts.reshape((-1, y_ts.shape[1], y_ts.shape[2], 1))
    # x_ts = np.array(x_ts, dtype=np.float64) / size
    # x_test, y_test = data_generator(1, size, 50, 0.5)
    pred = model.predict(img)
    # detected = pred * 200
    # params = y_test * 200
    #     print(detected)
    #     print(y_test*200)
    results.append(iou(params, pred))
results = np.array(results)
print((results > 0.7).sum())
#
# x_test, y_test = data_generator(1000, size, 50, 2)
# model = load_model(os.path.join('models', 'model_200.hdf5'), compile=False)
#
# pred = model.predict(x_test)
# detected = pred
# detected
# a = pred * 200
#
# b = y_test * 200
#
# results = []
# for i in range(1000):
#     results.append(iou(b[i], a[i]))
# results = np.array(results)
# print((results > 0.7).mean())
