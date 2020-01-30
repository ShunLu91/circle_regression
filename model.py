import os
import argparse
from main import *
from keras.optimizers import Adam
from keras.models import model_from_json, Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout, BatchNormalization


# read arguments from command
parser = argparse.ArgumentParser('Circle_Regression')
parser.add_argument('--epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch', type=int, default=64, help='train batch size')
parser.add_argument('--img_size', type=int, default=200, help='img_size')
parser.add_argument('--radius', type=int, default=50, help='radius')
parser.add_argument('--noise', type=int, default=2, help='noise')
parser.add_argument('--num', type=int, default=10000, help='num of training images')
args = parser.parse_args()
print(args)


def data_generator(samples, size, radius, noise):
    params = []
    images = []
    for _ in range(samples):
        para, img = noisy_circle(size, radius, noise)
        params.append(np.asarray(para))
        images.append(img)
    images = np.array(images, dtype=np.float)
    images = images.reshape((-1, images.shape[1], images.shape[2], 1))
    params = np.array(params, dtype=np.float64) / size
    params = np.array(params, dtype=np.float64)
    return images, params


def model_graph():
    # images = np.zeros((args.num, args.img_size, args.img_size), dtype=np.float)
    # # num circles in image = 1, number of outputs req'd = 3 (r,c,rad)
    # params = np.zeros((args.num, 1, 3), dtype=np.float)
    # for i in range(args.num):
    #     loc, images[i] = noisy_circle(img_size, rad, noise)
    #     params[i] = np.array(loc).reshape(1, 3)
    #
    # X = images.reshape(images.shape[0], 200, 200, 1)
    # Y = params.reshape(args.num, -1)  # /img_size #normalize to bring between 0 and 1

    images, params = data_generator(args.num, args.img_size, args.radius, args.noise)
    split = int(0.8 * args.num)
    Xtrain = images[:split]
    Xtest = images[split:]
    Ytrain = params[:split]
    Ytest = params[split:]

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
    cnn.add(Dense(params.shape[-1]))
    cnn.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mse', 'mae', 'cosine'])
    cnn.fit(Xtrain, Ytrain,
            batch_size=args.batch,
            epochs=args.epochs,
            verbose=1,
            validation_data=(Xtest, Ytest))

    return cnn


def find_circle(img, model):
    img = img.reshape(1, 200, 200, 1)
    pred = model.predict(img)
    pred = pred.reshape(-1)
    return pred


if __name__ ==  '__main__':
    if os.path.isfile('cnn_deeper.h5'):
        json_file = open('cnn_deeper.json', 'r')
        cnn_json = json_file.read()
        json_file.close()
        cnn = model_from_json(cnn_json)
        cnn.load_weights("cnn_deeper.h5")
    else:
        cnn = model_graph()
        cnn_json = cnn.to_json()
        with open("cnn_deeper.json", "w") as json_file:
            json_file.write(cnn_json)
        cnn.save_weights("cnn_deeper.h5")

    results = []
    for _ in range(1000):
        params, img = noisy_circle(args.img_size, args.radius, args.noise)
        detected = find_circle(img, cnn)
        print(iou(params, detected))
        results.append(iou(params, detected))
    results = np.array(results)
    print((results > 0.7).sum())
