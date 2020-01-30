import keras.layers as KL
from keras.models import Model, load_model


def network(size, filters, image_channels=1):
    inpt = KL.Input(shape=(size, size, image_channels), name='input_images')

    # stem
    x = KL.Conv2D(filters=filters, kernel_size=(3, 3), strides=2, padding='same', name='conv1')(inpt)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu', name='relu1')(x)

    x = KL.Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same', name='conv2')(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu', name='relu2')(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = KL.Conv2D(filters=filters * 2, kernel_size=(3, 3), strides=1, padding='same', name='conv3')(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu', name='relu3')(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = KL.Conv2D(filters=filters * 4, kernel_size=(3, 3), strides=1, padding='same', name='conv4')(x)
    x = KL.BatchNormalization()(x)
    x = KL.Activation('relu', name='relu4')(x)
    x = KL.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = KL.Flatten()(x)
    x = KL.Dense(1024, activation='relu')(x)
    x = KL.Dense(3)(x)
    model = Model(inputs=inpt, outputs=x)

    return model


def model_graph(size, filters, image_channels=1):
    inpt = KL.Input(shape=(size, size, image_channels), name='input_images')

    # stem
    x = KL.Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding='same', name='conv1')(inpt)
    x = KL.BatchNormalization(axis=-1, momentum=0.0, epsilon=0.0001, name='bn1')(x)
    x = KL.Activation('relu', name='relu1')(x)

    x = KL.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', name='conv2')(x)
    x = KL.Activation('relu', name='relu2')(x)
    #     x = KL.BatchNormalization(axis=-1, momentum=0.0,epsilon=0.0001, name = 'bn1')(x)
    # x = KL.MaxPooling2D(pool_size=(2, 2), strides=2)(x)

    x = KL.Conv2D(filters=64, kernel_size=(3, 3), strides=2, padding='same', name='conv3')(x)
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
    # x = KL.Dropout(0.5)(x)
    x = KL.Dense(3)(x)
    model = Model(inputs=inpt, outputs=x)

    return model
