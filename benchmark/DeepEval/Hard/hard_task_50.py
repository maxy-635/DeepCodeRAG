import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Lambda, MaxPooling2D, Flatten, Concatenate, SeparableConv2D, Dense, Dropout, Reshape

def dl_model():

    input_layer = Input(shape=(32,32,3))

    def block_1(input_tensor):
        
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)
        dropout1 = Dropout(rate=0.2)(flatten1)

        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)
        dropout2 = Dropout(rate=0.2)(flatten2)

        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)
        dropout3 = Dropout(rate=0.2)(flatten3)

        output_tensor = Concatenate()([dropout1, dropout2, dropout3])

        return output_tensor

    def block_2(input_tensor):
        
        inputs_groups = Lambda(function=lambda x: tf.split(value=x, num_or_size_splits=4, axis=-1))(input_tensor)
        conv1 = SeparableConv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs_groups[0])
        conv2 = SeparableConv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(inputs_groups[1])
        conv3 = SeparableConv2D(filters=64, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(inputs_groups[2])
        conv4 = SeparableConv2D(filters=64, kernel_size=(7, 7), strides=(1, 1), padding='same', activation='relu')(inputs_groups[3])
        output_tensor = Concatenate()([conv1, conv2, conv3, conv4])

        return output_tensor

    block_output = block_1(input_tensor=input_layer)
    dense_1 = Dense(units=256, activation='relu')(block_output)
    reshaped = Reshape(target_shape=(8, 8, 4))(dense_1)
    block_output2 = block_2(input_tensor=reshaped)

    flatten = Flatten()(block_output2)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model