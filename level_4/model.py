from keras.layers import MaxPool2D
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, BatchNormalization, Activation, AveragePooling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D Convolution-Batch Normalization-Activation stack builder
    # Arguments
        inputs (tensor): input tensor from input image or previous layer
        num_filters (int): Conv2D number of filters
        kernel_size (int): Conv2D square kernel dimensions
        strides (int): Conv2D square stride dimensions
        activation (string): activation name
        batch_normalization (bool): whether to include batch normalization
        conv_first (bool): conv-bn-activation (True) or
            bn-activation-conv (False)
    # Returns
        x (tensor): tensor as input to the next layer
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def get_resnet_v1(input_shape, depth, num_classes=10):
    """ResNet Version 1 Model builder [a]
    Stacks of 2 x (3 x 3) Conv2D-BN-ReLU
    Last ReLU is after the shortcut connection.
    At the beginning of each stage, the feature map size is halved (downsampled)
    by a convolutional layer with strides=2, while the number of filters is
    doubled. Within each stage, the layers have the same number filters and the
    same number of filters.
    Features maps sizes:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64
    The Number of parameters is approx the same as Table 6 of [a]:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M
    # Arguments
        input_shape (tensor): shape of input image tensor
        depth (int): number of core convolutional layers
        num_classes (int): number of classes (CIFAR10 has 10)
    # Returns
        model (Model): Keras model instance
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # Start model definition.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=7)(x)
    # x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_resnet_v1_20(shape, classes):
    shape = (shape[1], shape[2], shape[3])
    return get_resnet_v1(shape, 20, classes)


def get_vgg_mini(shape, classes):
    y_input = Input(shape=(shape[1], shape[2], shape[3]))

    layer = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(y_input)
    layer = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layer)
    layer = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    layer = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    layer = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    # layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layer)
    # layer = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    # layer = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    # layer = Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    # layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layer)
    # layer = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    # layer = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    # layer = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    # layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layer)
    # layer = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    # layer = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    # layer = Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu")(layer)
    # layer = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layer)

    layer = Flatten()(layer)
    layer = Dense(units=512, activation="relu")(layer)
    output_layer = Dense(units=classes, activation="sigmoid")(layer)

    new_model = Model(inputs=y_input, outputs=[output_layer], name='VGG-MINI')
    return new_model