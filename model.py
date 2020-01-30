import os
import argparse
from main import *
from keras.models import Model
from keras.optimizers import Adam
from keras.models import model_from_json, Sequential
from keras.layers import Input, Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization

# read arguments from command
parser = argparse.ArgumentParser('Circle_Regression')
parser.add_argument('--epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch', type=int, default=64, help='train batch size')
parser.add_argument('--img_size', type=int, default=200, help='img_size')
parser.add_argument('--radius', type=int, default=50, help='radius')
parser.add_argument('--noise', type=int, default=2, help='noise')
parser.add_argument('--num', type=int, default=10000, help='num of training images')
parser.add_argument('--spilt_rate', type=float, default=0.9, help='num of training images')
parser.add_argument('--seed', type=int, default=2020, help='num of training images')
args = parser.parse_args()
print(args)

# generate data
def data_generator(samples, size, radius, noise):
    params = []
    images = []
    for _ in range(samples):
        para, img = noisy_circle(size, radius, noise)
        params.append(np.asarray(para))
        images.append(img)
    images = np.array(images, dtype=np.float)
    images = images.reshape((-1, images.shape[1], images.shape[2], 1))
    params = np.array(params, dtype=np.float64)
    return images, params


def model_graph():
    input = Input(shape=(args.img_size, args.img_size, 1), name='input_images')
    x = Conv2D(64, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal')(input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(512, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Conv2D(512, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    y = Dense(3)(x)

    model = Model(inputs=input, outputs=y)
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mse', 'mae'])

    return model


def model_graph2():
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=(1, 1), data_format='channels_last',
                     input_shape=(args.img_size, args.img_size, 1), padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(128, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, kernel_size=3, strides=(1, 1), padding='same', kernel_initializer='glorot_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(3))
    model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mse', 'mae'])
    return model


def find_circle(img, model):
    img = img.reshape(1, 200, 200, 1)
    pred = model.predict(img)
    pred = pred.reshape(-1)
    return pred


if __name__ == '__main__':
    # set random seed
    np.random.seed(args.seed)
    images, params = data_generator(args.num, args.img_size, args.radius, args.noise)
    split = int(args.spilt_rate * args.num)
    img_train = images[:split]
    img_test = images[split:]
    param_train = params[:split]
    param_test = params[split:]
    model = model_graph()
    # model train
    model.fit(img_train, param_train, batch_size=args.batch, epochs=args.epochs, verbose=2,
              validation_data=(img_test, param_test))

    # model save
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("trained_model.h5")

    # evaluate
    results = []
    for _ in range(1000):
        params, img = noisy_circle(args.img_size, args.radius, args.noise)
        detected = find_circle(img, model)
        print(iou(params, detected))
        results.append(iou(params, detected))
    results = np.array(results)
    print('Num of IOU Images above 0.7:', (results > 0.7).sum())
