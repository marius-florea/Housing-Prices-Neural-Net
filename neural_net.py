
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import keras

import itertools

import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from processing_utils import *
from features import *
from utils import *

train_original = pd.read_csv('../data/train.csv', index_col='Id')
generated_df = pd.read_csv('syntetic_data_from_ae_less_rows_with_nans.csv', index_col='Id')

# print("train data duplicates",train_original.duplicated().values.sum())
# train_original = remove_rows_with_nans(train_original)


# train = train.select_dtypes(exclude=['object'])
# train.fillna(0,inplace=True)

test_original = pd.read_csv('../data/test.csv', index_col='Id')
# test = test_original.select_dtypes(exclude=['object'])
# test.fillna(0,inplace=True)

y_list =['SalePrice']
Y_train_original = train_original[y_list]
# train_original.drop(['SalePrice'], axis=1, inplace=True)

numerical_cols = get_high_corelated_numerical_features()
categorical_cols = get_choosen_categorical_features()

# "Cardinality" means the number of unique values in a column
# Select categorical co lumns with relatively low cardinality (convenient but arbitrary)
# categorical_cols = [cname for cname in train_original.columns if
#                     train_original[cname].nunique() < 15 and
#                     train_original[cname].dtype == "object"]

# Select numerical columns
# numerical_cols = [cname for cname in train_original.columns if
#                 train_original[cname].dtype in ['int64', 'float64']]
# numerical_cols.remove('SalePrice')

my_cols = numerical_cols + categorical_cols  #+ numeric_categorical_cols
my_cols_with_saleprice = my_cols + ['SalePrice']

train_na_filled = train_original.copy()
train_na_filled[numerical_cols] = train_na_filled[numerical_cols].fillna(-1)#see if does same as line below
test_na_filled = test_original.copy()
test_na_filled[numerical_cols] = test_na_filled[numerical_cols].fillna(-1)


#commentig the bellow line gives even a better score wtf?
train_na_filled = train_na_filled.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('inexistent'))
test_na_filled = test_na_filled.apply(lambda x: x.fillna(0) if x.dtype.kind in 'biufc' else x.fillna('inexistent'))

#kept only the selected columns adica my_cols_with_salesprice
train_na_filled = train_na_filled[my_cols_with_saleprice].copy()
test_na_filled = test_na_filled[my_cols].copy()

from processing_utils import *
preprocessor_train = get_preprocessor(numerical_cols, categorical_cols, y_list)
preprocessor_test = get_preprocessor(numerical_cols, categorical_cols)

processed_X_full = pd.DataFrame(preprocessor_train.fit_transform(train_na_filled))
processed_X_full.columns = train_na_filled.columns

frame = pd.concat([processed_X_full,generated_df], axis=0, ignore_index=True)
processed_X_full = frame
processed_X_full = processed_X_full.sample(frac=1).reset_index(drop=True)


processed_X_test = pd.DataFrame(preprocessor_test.fit_transform(test_na_filled))
processed_X_test.columns = test_na_filled.columns


# X_train_final = processed_X_full
# print(X_train_final.shape)
# print(train_modified.shape)




# In this small part we will isolate the outliers with an IsolationFores
from sklearn.ensemble import IsolationForest

clf = IsolationForest(max_samples=100, random_state=42)
clf.fit(processed_X_full)
y_noano = clf.predict(processed_X_full)
y_noano = pd.DataFrame(y_noano, columns=['Top'])
# y_noano[y_noano['Top'] == 1].index.values

train = processed_X_full.iloc[y_noano[y_noano['Top'] == 1].index.values]
test = processed_X_test
train.reset_index(drop=True, inplace=True)
print("number of outliers:",y_noano[y_noano['Top'] == -1].shape[0])
print("number of rows without outliers",train.shape[0])

print(train.head(10))

#Preprocessing To rescale our data we will use the fonction MinMaxScaler of Scikit-learn

import  warnings
# warnings.filterwarnings('ignore')

col_train = list(train.columns)
col_train_bis = list(train.columns)
col_train_bis.remove('SalePrice')

mat_train = np.matrix(train)
mat_test = np.matrix(test)
mat_new = np.matrix(train.drop('SalePrice', axis=1))
mat_y = np.array(train.SalePrice).reshape((train.shape[0],1))

prepro_y = MinMaxScaler()
prepro_y.fit(mat_y)

prepro = MinMaxScaler()
prepro.fit(mat_train)

prepro_test = MinMaxScaler()
prepro_test.fit(mat_test)# or prepro_test.fit(mat_test) ??

train = pd.DataFrame(prepro.transform(mat_train), columns=col_train)
test = pd.DataFrame(prepro_test.transform(mat_test),columns=col_train_bis)

#minimized syntetic data
# train_syntetic_data = pd.read_csv('minimized_predictions_from_ae.csv', index_col='Id')
# train = pd.concat([train,train_syntetic_data])


print(train.head())

# List of features
COLUMNS = col_train
FEATURES = col_train_bis
LABEL = 'SalePrice'

#columns
feature_cols = FEATURES

#Training set and Prediction set with the features tor predict
training_set = train[COLUMNS]
y_label = train.SalePrice

#Train and Test

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import *
from keras.wrappers.scikit_learn import KerasRegressor
from keras.losses import  *

#model
# dropout_rate0 = 0.05
# dropout_rate1 = 0.2
dropout_rate2 = 0.05
dropout_rate3 = 0.1

import time
# ts stores the time in seconds
ts = time.time()
textfile_name = "Loses/"+str(ts) + ".txt"

from tensorflow.keras import regularizers
import tensorflow as tf

def root_mean_squared_error(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true)))

def model_function(input_dimension,optimizer="adam",instantiate=False):
    def create_model():
        l1 = 9*1e-4
        l2 = 5*1e-4
        kernel_initializer = 'glorot_uniform'
        model = Sequential()
        model.add(Dense(input_dimension, input_dim=input_dimension, kernel_initializer=kernel_initializer, activation='relu'))
        # model.add(Dense(130, kernel_initializer=kernel_initializer, kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2),activation='relu'))
        # model.add(Dropout(dropout_rate2))
        # model.add(Dense(120, kernel_initializer=kernel_initializer,kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2), activation='relu'))
        model.add(Dropout(dropout_rate2))
        model.add(Dense(100, kernel_initializer=kernel_initializer,kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2), activation='relu'))
        model.add(Dropout(dropout_rate2))
        model.add(Dense(65, kernel_initializer=kernel_initializer,kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2), activation='relu'))
        model.add(Dropout(dropout_rate2))
        # model.add(Dense(40, kernel_initializer=kernel_initializer,kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2), activation='relu'))
        # model.add(Dense(20, kernel_initializer=kernel_initializer,kernel_regularizer=regularizers.l1_l2(l1=l1,l2=l2), activation='relu'))
        model.add(Dense(1, kernel_initializer=kernel_initializer))
        #compile model
        model.compile(metrics=['accuracy'],loss='binary_crossentropy',optimizer=optimizer)
        return model
    function = create_model() if instantiate else create_model
    return function



from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
strat_kfold = StratifiedKFold()

training_set_selection = training_set[FEATURES]

x_train_cross_val, x_holdout_test, y_train_cross_val, y_test = train_test_split(training_set_selection, y_label,
                                                                                test_size=0.15, random_state=42)
#need to pass the model and return the fitted model
def manual_cv(x_train_cross_val, y_train_cross_val, model,epochs=120,batch_size=100):

    train_loses = np.zeros(0)
    cv_losses = np.zeros(0)
    # for i in range(2):#wtf delete or comment
    i=1
    kf = KFold(n_splits=4, random_state=np.random.randint(10000*(9-i)))
    for train_index, test_index in kf.split(x_train_cross_val,y_train_cross_val):
        x_train, x_validation = training_set_selection.iloc[train_index], training_set_selection.iloc[test_index]
        y_train, y_validation = y_label.iloc[train_index], y_label.iloc[test_index]

        # y_train = pd.DataFrame(y_train, columns=[LABEL])
        # training_set = pd.DataFrame(x_train, columns= FEATURES).merge(y_train, left_index=True,
        #                                                           right_index=True)
        #
        # # Training for submission
        # training_sub = training_set[col_train]
        #
        # # Same thing but for the test set
        y_validation = pd.DataFrame(y_validation, columns= [LABEL])
        validation_set = pd.DataFrame(x_validation, columns=FEATURES).merge(y_validation, left_index=True,
                                                                            right_index=True)
        # print(validation_set.head())


        # feature_cols = training_set[FEATURES]
        # labels = training_set[LABEL].values

        model.fit(np.array(x_train),np.array(y_train), epochs=epochs, batch_size=batch_size,verbose=0)
        loss = model.evaluate(np.array(x_train), np.array(y_train))
        print(model.metrics_names,":",loss)
        train_loses = np.append(train_loses,loss[0])
        # Predictions
        # feature_cols_test = validation_set[FEATURES]
        # labels_test = validation_set[LABEL].values

        # y = model.predict(np.array(x_validation))
        # predictions = list(itertools.islice(y, x_validation.shape[0]))

        validation_data_loss = model.evaluate(np.array(x_validation),np.array(y_validation))
        print("validation",model.metrics_names,validation_data_loss)
        cv_losses = np.append(cv_losses,validation_data_loss[0])

        plot = False
        if plot:
            predictions = prepro_y.inverse_transform(np.array(predictions).reshape(len(predictions),1))
            reality = pd.DataFrame(prepro.inverse_transform(validation_set), columns=np.array(COLUMNS)).SalePrice

            matplotlib.rc('xtick', labelsize=10)
            matplotlib.rc('ytick', labelsize=10)

            fig, ax = plt.subplots(figsize=(10,10))
            plt.style.use('ggplot')
            plt.plot(predictions, reality, 'ro')
            plt.xlabel('Predictions', fontsize=10)
            plt.ylabel('Reality', fontsize=10)

            plt.title('Predictions x Reality on dataset Test', fontsize = 30)
            ax.plot([reality.min(), reality.max()], [reality.min(), reality.max()], 'k--', lw=4)
            plt.show()
            plt.close()
            # print()

    cv_losses_mean = cv_losses.mean()
    train_loses_string = "train losses " + str(train_loses)
    cv_losses_string = "cv_loses " + str(cv_losses)
    print(train_loses_string)
    print(cv_losses_string)

    textfile = open(textfile_name, "w")
    textfile.writelines([train_loses_string,"\n",cv_losses_string,"\n"])
    textfile.writelines(["cv_loss mean",str(cv_losses_mean)])
    textfile.writelines(["epochs",str(epochs)])
    textfile.close()

    print("cv loss mean:",cv_losses_mean)
    return model

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import make_scorer

def grid_cv(x_train_cross_val,y_train_cross_val,param_grid, cv=5,scoring_fit=root_mean_squared_error):

    scorer = make_scorer(scoring_fit, greater_is_better=False)
    input_dim = len(feature_cols)
    kerasRegressor = KerasRegressor(build_fn=model_function(input_dimension=input_dim))
    gs = GridSearchCV(estimator=kerasRegressor,
                      param_grid=param_grid,
                      cv=cv,
                      n_jobs=-1,
                      # scoring=scorer,
                      verbose=2)
    fitted_model = gs.fit(x_train_cross_val,y_train_cross_val)
    print("gs best params",gs.best_params_)
    print("gs best score", gs.best_score_)

    return fitted_model

epochs = 300
param_grid = {
              'epochs':[100,120,130],
              'batch_size':[50,100],
              # 'optimizer':['Adam']
              # 'dropout_rate' : [0.0, 0.1, 0.2],
              # 'activation' :          ['relu', 'elu']
             }

do_manual_cv = True
fitted_model: Sequential

if do_manual_cv:
    input_dim = len(feature_cols)
    model = model_function(input_dimension=input_dim,instantiate=True)
    fitted_model = manual_cv(x_train_cross_val,y_train_cross_val,model,epochs=epochs,batch_size=50)
else:
    fitted_model = grid_cv(x_train_cross_val,y_train_cross_val,param_grid,cv=5)

# loss = fitted_model.evaluate(np.array(x_test), np.array(y_test))
y_prediction_test = fitted_model.predict(np.array(x_holdout_test))
# loss = mean_absolute_error(np.array(y_test),y_prediction_test)
loss = binary_crossentropy(np.array(y_test),y_prediction_test)
loss_on_holdoutset_text = ""
if do_manual_cv:
    loss_manual = loss.numpy().mean()
    loss_on_holdoutset_text = "manualcv loss on the hold out final test set:" + str(loss_manual)
else:
    loss_on_holdoutset_text = "gridsearchcv loss on the hold out final test set:" + str(loss)

print(loss_on_holdoutset_text)
with open(textfile_name,'a') as f:
    f.write(loss_on_holdoutset_text+"\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))


# prediction for sumbmission
y_predict = fitted_model.predict(np.array(test))

def to_submit(pred_y, name_out):
    y_predict = list(itertools.islice(pred_y,test.shape[0]))
    y_predict = pd.DataFrame(prepro_y.inverse_transform(np.array(y_predict).
                                                        reshape(len(y_predict),1)),
                             columns=['SalePrice'])
    # y_predict = y_predict.join(ID)
    output = pd.DataFrame({'Id': test_original.index,
                           'SalePrice': y_predict.SalePrice})

    output.to_csv(name_out+'.csv', index=False)

to_submit(y_predict,"submission")

