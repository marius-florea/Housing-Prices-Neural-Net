import keras
from keras import layers
from features import *
import numpy as np
import pandas as pd
from keras.losses import *
encoding_dim = 32
input_size = 65
input_data = keras.Input(shape=(input_size,))
encoded = layers.Dense(encoding_dim,activation='relu')(input_data)
decoded = layers.Dense(input_size, activation='sigmoid')(encoded)

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
arr = np.array(processed_X_full)
autoencoder.fit(arr, arr, epochs=51, batch_size=50, shuffle=True)
pp = autoencoder.predict(processed_X_full)
print("")

