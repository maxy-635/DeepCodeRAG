from keras.models import Model
from keras.layers import Input, MaxPooling2D, Flatten, Conv2D, Concatenate, Dense, Reshape

def dl_model():
    
    input_layer = Input(shape=(28, 28, 1))

    def block_1(input_tensor):
        
        maxpool1 = MaxPooling2D(pool_size=(1, 1), strides=(1, 1), padding='same')(input_tensor)
        flatten1 = Flatten()(maxpool1)

        maxpool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(input_tensor)
        flatten2 = Flatten()(maxpool2)

        maxpool3 = MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')(input_tensor)
        flatten3 = Flatten()(maxpool3)

        output_tensor = Concatenate()([flatten1, flatten2, flatten3])

        return output_tensor

    def block_2(input_tensor):

        conv1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(2, 2), padding='same', activation='relu')(input_tensor)
        conv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding='same', activation='relu')(input_tensor)
        conv3 = Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='relu')(input_tensor)
        maxpool = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(input_tensor)
        output_tensor = Concatenate()([conv1, conv2, conv3, maxpool])

        return output_tensor

    block_output1 = block_1(input_tensor=input_layer)
    dense_block1 = Dense(units=256, activation='relu')(block_output1)
    reshaped = Reshape(target_shape=(1, 1, 256))(dense_block1)
    block_output2 = block_2(input_tensor=reshaped)

    flatten = Flatten()(block_output2)
    output_layer = Dense(units=10, activation='softmax')(flatten)
    
    model = Model(inputs=input_layer, outputs=output_layer)

    return model
