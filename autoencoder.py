import keras
from keras import layers
from features import *
import numpy as np
import pandas as pd
from keras.losses import *
import tensorflow

print("tensorflow",tensorflow.version.VERSION)
print("keras version",keras.__version__)

encoding_dim = 32
input_size = 65
input_data = keras.Input(shape=(65,))
print("inpuyt data",input_data)

encoded = layers.Dense(input_size,input_dim=input_size)(input_data)
encoded = layers.Dense(260,activation='relu')(encoded)
encoded = layers.Dense(130,activation='relu')(encoded)

decoded = layers.Dense(130, activation='relu')(encoded)
decoded = layers.Dense(260, activation='relu')(decoded)
decoded = layers.Dense(input_size, activation='relu')(decoded)

autoencoder = keras.Model(input_data, decoded)

autoencoder.compile(optimizer='adam', loss=mean_squared_logarithmic_error)

train_original = pd.read_csv('../data/train.csv', index_col='Id')

numerical_cols = get_high_corelated_numerical_features()
categorical_cols = get_choosen_categorical_features()

my_cols = numerical_cols + categorical_cols  #+ numeric_categorical_cols
my_cols_with_saleprice = my_cols + ['SalePrice']

train_na_filled = train_original.copy()
train_na_filled[numerical_cols] = train_na_filled[numerical_cols].fillna(-1)#see if does same as line below


#commentig the bellow line gives even a better score wtf?
train_na_filled = train_na_filled.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('inexistent'))

#kept only the selected columns adica my_cols_with_salesprice
train_na_filled = train_na_filled[my_cols_with_saleprice].copy()

from processing_utils import *
y_list =['SalePrice']
preprocessor_train = get_preprocessor(numerical_cols, categorical_cols, y_list)

processed_X_full = pd.DataFrame(preprocessor_train.fit_transform(train_na_filled))
processed_X_full.columns = train_na_filled.columns
print("")
from sklearn.preprocessing import MinMaxScaler

mat_train = np.array(processed_X_full)
min_max_scaler = MinMaxScaler()
min_max_scaler.fit(mat_train)
col_train = list(processed_X_full.columns)

mat_train_scaled = pd.DataFrame(min_max_scaler.transform(mat_train), columns=col_train)

# arr = np.expand_dims(arr, axis=2)
autoencoder.fit(mat_train_scaled, mat_train_scaled, epochs=2000, batch_size=100, shuffle=True,validation_data=(mat_train_scaled,mat_train_scaled))
mat_train_prediction = autoencoder.predict(mat_train_scaled)
prediction_df = pd.DataFrame(mat_train_prediction)
prediction_df.columns = processed_X_full.columns


print("")

