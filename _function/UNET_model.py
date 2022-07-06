from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate, Dropout, MaxPooling2D, Input, BatchNormalization
from tensorflow.keras.models import Model

def UNET(num_classes, shape):

    """ ENCODE_BLOCK """

    INPUT = Input(shape)

    conv_1_down = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation = 'relu', kernel_initializer = 'he_normal')(INPUT)
    conv_1_down = BatchNormalization()(conv_1_down)
    conv_1_down = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation = 'relu', kernel_initializer = 'he_normal')(conv_1_down)
    conv_1_down = BatchNormalization()(conv_1_down)

    maxpool1 = MaxPooling2D(pool_size=2)(conv_1_down)
    conv_2_down = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation = 'relu', kernel_initializer = 'he_normal')(maxpool1)
    conv_2_down = BatchNormalization()(conv_2_down)
    conv_2_down = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', activation = 'relu', kernel_initializer = 'he_normal')(conv_2_down)
    conv_2_down = BatchNormalization()(conv_2_down)
    
    maxpool2 = MaxPooling2D(pool_size=2)(conv_2_down)
    conv_3_down = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation = 'relu', kernel_initializer = 'he_normal')(maxpool2)
    conv_3_down = BatchNormalization()(conv_3_down)
    conv_3_down = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', activation = 'relu', kernel_initializer = 'he_normal')(conv_3_down)
    conv_3_down = BatchNormalization()(conv_3_down)
    
    maxpool3 = MaxPooling2D(pool_size=2)(conv_3_down)
    conv_4_down = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation = 'relu', kernel_initializer = 'he_normal')(maxpool3)
    conv_4_down = BatchNormalization()(conv_4_down)
    conv_4_down = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', activation = 'relu', kernel_initializer = 'he_normal')(conv_4_down)
    conv_4_down = BatchNormalization()(conv_4_down)
    
    maxpool4 = MaxPooling2D(pool_size=2)(conv_4_down)
    conv_5_down = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation = 'relu', kernel_initializer = 'he_normal')(maxpool4)
    conv_5_down = BatchNormalization()(conv_5_down)
    conv_5_down = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', activation = 'relu', kernel_initializer = 'he_normal')(conv_5_down)
    conv_5_down = BatchNormalization()(conv_5_down)

    drop = Dropout(0.5)(conv_5_down)

    """ DECODE_BLOCK """

    conv_4_up = Conv2D(256, 2, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop))
    concat1 = Concatenate()([conv_4_down, conv_4_up])
    conv_4_up = Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(concat1)
    conv_4_up = BatchNormalization()(conv_4_up)
    conv_4_up = Conv2D(256, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(conv_4_up)
    conv_4_up = BatchNormalization()(conv_4_up)

    conv_3_up = Conv2D(128, 2, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_4_up))
    concat2 = Concatenate()([conv_3_up, conv_3_down])
    conv_3_up = Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(concat2)
    conv_3_up = BatchNormalization()(conv_3_up)
    conv_3_up = Conv2D(128, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(conv_3_up)
    conv_3_up = BatchNormalization()(conv_3_up)

    conv_2_up = Conv2D(64, 2, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_3_up))
    concat3 = Concatenate()([conv_2_up, conv_2_down])
    conv_2_up = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(concat3)
    conv_2_up = BatchNormalization()(conv_2_up)
    conv_2_up = Conv2D(64, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(conv_2_up)
    conv_2_up = BatchNormalization()(conv_2_up)

    conv_1_up = Conv2D(32, 2, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv_2_up))
    concat4 = Concatenate()([conv_1_up, conv_1_down])
    conv_1_up = Conv2D(32, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(concat4)
    conv_1_up = BatchNormalization()(conv_1_up)
    conv_1_up = Conv2D(32, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(conv_1_up)
    conv_1_up = BatchNormalization()(conv_1_up)

    conv_1_up = Conv2D(32, 3, padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(conv_1_up)

    OUTPUT = Conv2D(num_classes, (1, 1), activation='sigmoid')(conv_1_up)

    model = Model(inputs=INPUT, outputs=OUTPUT)

    return model