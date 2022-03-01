from tensorflow.keras.models import  Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import *
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.losses import *
import tensorflow.keras as keras
from tensorflow.keras import regularizers

#model
# dropout_rate0 = 0.05
# dropout_rate1 = 0.2
dropout_rate2 = 0.05
dropout_rate3 = 0.1

def model_function(input_dimension,optimizer=keras.optimizers.Adam(learning_rate=0.002),instantiate=False):
    def create_model():
        l1 = 9*1e-4
        l2 = 5*1e-4
        kernel_initializer = 'glorot_uniform'
        model = Sequential()
        # input_shape = (input_dim,1)
        model.add(Dense(input_dimension,input_dim=input_dimension, kernel_initializer=kernel_initializer, activation='relu'))

        #uncomment --->
        model.add(Dropout(dropout_rate2))
        model.add(Dense(100, kernel_initializer=kernel_initializer,kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2), activation='relu'))
        model.add(Dropout(dropout_rate2))
        model.add(Dense(65, kernel_initializer=kernel_initializer,kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2), activation='relu'))
        model.add(Dropout(dropout_rate2))
        # <---- uncomment

        # model.add(Dense(50, kernel_initializer=kernel_initializer,kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2), activation='relu'))
        model.add(Dense(1, kernel_initializer=kernel_initializer))
        #compile model
        model.compile(metrics=['accuracy'],loss='binary_crossentropy',optimizer=optimizer)
        return model
    function = create_model() if instantiate else create_model
    return function

def model_function_large(input_dimension, learning_rate=0.001, instantiate=False):
    def create_model(input_dimension, learning_rate):
        optimizer = keras.optimizers.Adam(lr=learning_rate)
        l1 = 1*1e-3
        l2 = 1*1e-3

        kernel_initializer = 'glorot_uniform'
        model = Sequential()
        # input_shape = (input_dim,1)
        model.add(Dense(50,input_dim=input_dimension, kernel_initializer=kernel_initializer))

        #uncomment --->
        model.add(Dropout(dropout_rate2))
        model.add(Dense(100, kernel_initializer=kernel_initializer,kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2), activation='relu'))
        model.add(Dropout(dropout_rate2))
        model.add(Dense(50, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
                        activation='relu'))
        # model.add(Dropout(dropout_rate2))
        # model.add(Dense(100, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1=l1, l2=l2),
        #                 activation='relu'))
        # model.add(Dropout(dropout_rate2))
        # model.add(Dense(65, kernel_initializer=kernel_initializer,kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2), activation='relu'))
        # model.add(Dropout(dropout_rate2))
        # <---- uncomment

        model.add(Dense(1, kernel_initializer=kernel_initializer))
        #compile model
        model.compile(metrics=['accuracy'],loss='binary_crossentropy',optimizer=optimizer)
        return model
    function = create_model(input_dimension=input_dimension,learning_rate=learning_rate) if instantiate else create_model
    return function
