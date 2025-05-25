import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, GRU, Masking, concatenate


def transfer(model_path, ntimesteps, n_features, contextual, latent_size, old_output, new_output, initial = False):
    input_1 = Input((ntimesteps, n_features), name='lc')  # X.shape = (Nobjects, Ntimesteps, 4) CHANGE
    masking_input1 = Masking(mask_value=0.)(input_1)

    lstm1 = GRU(100, return_sequences=True, activation='tanh', trainable=initial, name='gru_1')(masking_input1)
    lstm2 = GRU(100, return_sequences=False, activation='tanh', trainable=initial, name='gru_2')(lstm1)

    dense1 = Dense(100, activation='tanh', name='dense_1')(lstm2)

    if (contextual == 0):
        merge1 = dense1
    else:
        input_2 = Input(shape = (contextual, ), name='host') # CHANGE
        dense2 = Dense(10, trainable=False, name='dense_2')(input_2)
        merge1 = concatenate([dense1, dense2])

    dense3 = Dense(100, activation='relu', trainable=False, name='dense_3')(merge1)

    dense4 = Dense(latent_size, activation='relu', name='latent', trainable=True)(dense3)

    output = Dense(old_output, activation='softmax', name='old_output')(dense4)

    if (contextual == 0):
        model = keras.Model(inputs=input_1, outputs=output)
    else:
        model = keras.Model(inputs=[input_1, input_2], outputs=output)

    model.load_weights(model_path)

    new_output = Dense(new_output, activation='softmax', name='output')(dense4)
    if (contextual == 0):
        model = keras.Model(inputs=input_1, outputs=new_output)
    else:
        model = keras.Model(inputs=[input_1, input_2], outputs=new_output)
    
    model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    return model

def build_model(latent_size, ntimesteps, num_classes, contextual, n_features=4):
    input_1 = Input((ntimesteps, n_features), name='lc')  # X.shape = (Nobjects, Ntimesteps, 4) CHANGE
    masking_input1 = Masking(mask_value=0.)(input_1)

    lstm1 = GRU(100, return_sequences=True, activation='tanh', name='gru_1')(masking_input1)
    lstm2 = GRU(100, return_sequences=False, activation='tanh', name='gru_2')(lstm1)

    dense1 = Dense(100, activation='tanh', name='dense_1')(lstm2)

    if (contextual == 0):
        merge1 = dense1
    else:
        input_2 = Input(shape = (contextual, ), name='host') # CHANGE
        dense2 = Dense(10, name='dense_2')(input_2)
        merge1 = concatenate([dense1, dense2])

    dense3 = Dense(100, activation='relu', name='dense_3')(merge1)

    dense4 = Dense(latent_size, activation='relu', name='latent')(dense3)

    output = Dense(num_classes, activation='softmax', name='output')(dense4)

    if (contextual == 0):
        model = keras.Model(inputs=input_1, outputs=output)
    else:
        model = keras.Model(inputs=[input_1, input_2], outputs=output)

    model.compile(loss = "categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    return model