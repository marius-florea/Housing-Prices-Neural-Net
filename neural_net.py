from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from tensorflow import keras

import itertools

import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from utils import *
import numpy as np
from preprocessing_dataframes import *

# train_original = remove_rows_with_nans(train_original)
# differentiate_NA_and_nans()
na_values = ["", "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN", "<NA>", "N/A", "NULL", "NaN", "n/a", "nan", "null"]
#these csv's can;t be read well in open office after the processing
train_original = pd.read_csv('processed data/train_processed.csv', index_col='Id',na_values=na_values, keep_default_na=False)
test_original = pd.read_csv('processed data/test_processed.csv', index_col='Id',na_values=na_values, keep_default_na=False)

processed_X_full, processed_X_test = load_and_preprocess_dataframes(train_original, test_original)

#TODO maybe remove at some point
# processed_X_full.to_csv("processed data/train_proccesed_w_dummies.csv")
# processed_X_test.to_csv("processed data/test_processed_w_dummies.csv")

# processed_X_full.columns = train_na_filled.columns
# processed_X_test = pd.DataFrame(preprocessor_test.fit_transform(test_na_filled))
# processed_X_test.columns = test_na_filled.columns

#This code computes differences of values present in train or test
#differences that can be found when from columns wiht different unique values are
#computed into one_hot encoded cols
#------------------------------------ Begining test code ---------------------------------------------------------------
processed_X_full_columns = processed_X_full.columns.to_list()
processed_X_test_columns = processed_X_test.columns.to_list()
processed_X_full_columns_set = set(processed_X_full_columns)
processed_X_test_columns_set = set(processed_X_test_columns)
in_train_not_in_test = (processed_X_full_columns_set - processed_X_test_columns_set)
in_test_not_in_train = (processed_X_test_columns_set - processed_X_full_columns_set)
#------------------------------------ End of test code ---------------------------------------------------------------
in_train_not_in_test.remove('SalePrice')
processed_X_full.drop(columns=in_train_not_in_test, inplace=True)

from sklearn.feature_selection import VarianceThreshold
variance_threshold = 0.001
if variance_threshold > 0.0 :
    varianceThreshold = VarianceThreshold(variance_threshold)
    processed_X_full_selection_arr = varianceThreshold.fit_transform(processed_X_full)
    print("new shape ",processed_X_full_selection_arr.shape)
    print(processed_X_full.shape)
    processed_X_full_selection = pd.DataFrame(processed_X_full_selection_arr,columns=varianceThreshold.get_feature_names_out())
    processed_X_full = processed_X_full_selection
    #TODO: here the cols for test were choosen based on the cols from train
    #this could have some bias pls verify
    variance_selection_train_columns = list(processed_X_full.columns)
    variance_selection_train_columns.remove('SalePrice')
    variance_selection_test_columns = variance_selection_train_columns
    processed_X_test = processed_X_test[variance_selection_test_columns]

# X_train_final = processed_X_full
# print(X_train_final.shape)
# print(train_modified.shape)

# In this small part we will isolate the outliers with an IsolationFores
isolate = True
from sklearn.ensemble import IsolationForest

def isolated_set(df: pd.DataFrame):
    clf = IsolationForest(n_estimators=200, max_samples=int(df.shape[0]/2),random_state=42)
    clf.fit(df)
    y_noano = clf.predict(df)
    y_noano = pd.DataFrame(y_noano, columns=['Include'])
    # y_noano[y_noano['Top'] == 1].index.values
    df = df.iloc[y_noano[y_noano['Include'] == 1].index.values]
    return df

# if isolate:
#     train = isolated_set(processed_X_full)
# else:
#     train = processed_X_full

train = processed_X_full #ori codu cu isolation forest ori linia asta!!!!\
test = processed_X_test
# train.reset_index(drop=True, inplace=True)
# print("number of outliers:",y_noano[y_noano['Top'] == -1].shape[0])
# print("number of rows without outliers",train.shape[0])

print(train.head(10))

#Preprocessing To rescale our data we will use the fonction MinMaxScaler of Scikit-learn

import  warnings
# warnings.filterwarnings('ignore')

test_columns_length = test.columns.__len__()
col_train = list(train.columns)
col_train_bis = list(train.columns)
col_train_bis.remove('SalePrice')
columns_from_test = list(test.columns)

mat_train = np.matrix(train)
mat_test = np.matrix(test)
mat_new = np.matrix(train.drop('SalePrice', axis=1))
mat_y = np.array(train.SalePrice).reshape((train.shape[0],1))

minmax_scaler_y = MinMaxScaler()
minmax_scaler_y.fit(mat_y)

minmax_scaler_train = MinMaxScaler()
minmax_scaler_train.fit(mat_train)

minmax_scaler_test = MinMaxScaler()
minmax_scaler_test.fit(mat_test)# or prepro_test.fit(mat_test) ??

train = pd.DataFrame(minmax_scaler_train.transform(mat_train), columns=col_train)

#add/remove these lines for syntetic data
# generated_df = pd.read_csv('syntetic_data_from_ae_2.csv', index_col='Id')
# frame = pd.concat([train,generated_df], axis=0, ignore_index=True)
# train = frame

test = pd.DataFrame(minmax_scaler_test.transform(mat_test), columns=columns_from_test)

# train.to_csv("train_minmax.csv")
# test.to_csv("test_minmax.csv")

random_state_nr = 42
train = train.sample(frac=1, random_state=random_state_nr).reset_index(drop=True)

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

# from tensorflow.keras.models import  Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import *
# from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
# from tensorflow.keras.losses import *


import time
# ts stores the time in seconds
ts = time.time()
textfile_name = "Loses/"+str(ts) + ".txt"

import tensorflow as tf

def root_mean_squared_error(y_true, y_pred):
    return tf.math.sqrt(tf.math.reduce_mean(tf.math.square(y_pred - y_true)))

from nn_model import *
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
strat_kfold = StratifiedKFold()

training_set_selection = training_set[FEATURES]

x_train_cross_val, x_holdout_set, y_train_cross_val, y_holdout_set = train_test_split(training_set_selection, y_label,
                                                                                      test_size=0.15, random_state=42)
def get_dataset_from_pd_dataframe(x:pd.DataFrame, y:pd.DataFrame):
    dataset = (
        tf.data.Dataset.from_tensor_slices(
            (
               tf.convert_to_tensor(x.to_numpy()),
               tf.convert_to_tensor(y.to_numpy())
            )
        )
    )
    dataset = dataset.batch(batch_size=80)
    return dataset

from tensorflow.keras.callbacks import EarlyStopping
from CustomStopper import *

def manual_cv_with_tfdataset(x_train_cross_val, y_train_cross_val, model,
                             epochs=120, batch_size=100, patience=100):
    with open(textfile_name, 'a') as f:
        f.write("\n manual_cv_with_tfdataset patience:  " + str(patience))

    train_loses = np.zeros(0)
    cv_losses = np.zeros(0)
    # for i in range(2):#wtf delete or comment was like this because gridsearch after finding the
    # good params reruns the model on data again
    i = 1
    kf = KFold(n_splits=4)  # , random_state=np.random.randint(10000*(9-i)))
    earlyStopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=patience)
    customStopper = CustomStopper(monitor='val_loss', mode='min',verbose=1,patience=patience,start_epoch=200)
    # print(tf.__version__)
    # print(tf.config.list_logical_devices())
    for train_index, test_index in kf.split(x_train_cross_val, y_train_cross_val):
        x_train, x_validation = x_train_cross_val.iloc[train_index], x_train_cross_val.iloc[test_index]
        y_train, y_validation = y_train_cross_val.iloc[train_index], y_train_cross_val.iloc[test_index]

        train_dataset = get_dataset_from_pd_dataframe(x_train, y_train)
        validation_dataset = get_dataset_from_pd_dataframe(x_validation, y_validation)

        model.fit(x=train_dataset, epochs=epochs,validation_data=validation_dataset, verbose=0,
                  callbacks=[customStopper])
        loss = model.evaluate(np.array(x_train), np.array(y_train))
        print(model.metrics_names, ":", loss)
        train_loses = np.append(train_loses, loss[0])

        validation_data_loss = model.evaluate(np.array(x_validation), np.array(y_validation))
        print("validation", model.metrics_names, validation_data_loss)
        cv_losses = np.append(cv_losses, validation_data_loss[0])

    cv_losses_mean = cv_losses.mean()
    train_loses_string = "train losses " + str(train_loses)
    cv_losses_string = "cv_loses " + str(cv_losses)
    print(train_loses_string)
    print(cv_losses_string)

    textfile = open(textfile_name, "a")
    textfile.writelines([train_loses_string, "\n", cv_losses_string])
    textfile.writelines(["\n cv_loss mean", str(cv_losses_mean)])
    textfile.writelines(["\n uniqeue margin", str(unique_margin)])
    textfile.writelines(["\n variance threshold", str(variance_threshold)])
    textfile.close()

    print("cv loss mean:", cv_losses_mean)
    return model

#need to pass the model and return the fitted model
def manual_cv(x_train_cross_val, y_train_cross_val, model,epochs=120,batch_size=100):
    with open(textfile_name, 'a') as f:
        f.write("\n manual_cv ")
    train_loses = np.zeros(0)
    cv_losses = np.zeros(0)
    # for i in range(2):#wtf delete or comment
    i=1
    kf = KFold(n_splits=4)#, random_state=np.random.randint(10000*(9-i)))
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
            predictions = minmax_scaler_y.inverse_transform(np.array(predictions).reshape(len(predictions), 1))
            reality = pd.DataFrame(minmax_scaler_train.inverse_transform(validation_set), columns=np.array(COLUMNS)).SalePrice

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
    textfile.writelines([train_loses_string,"\n",cv_losses_string])
    textfile.writelines(["\n cv_loss mean",str(cv_losses_mean)])
    textfile.writelines(["\n epochs",str(epochs)])
    textfile.writelines(["\n uniqeue margin",str(unique_margin)])
    textfile.writelines(["\n variance threshold",str(variance_threshold)])
    textfile.close()

    print("cv loss mean:",cv_losses_mean)
    return model

from sklearn.model_selection import GridSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import make_scorer

def grid_cv(x_train_cross_val,y_train_cross_val,param_grid,learning_rate , cv=5,scoring_fit=root_mean_squared_error):
    with open(textfile_name, 'a') as f:
        f.write("\n manual_cv ")

    scorer = make_scorer(scoring_fit, greater_is_better=False)
    input_dim = len(feature_cols)
    kerasRegressor = KerasRegressor(build_fn=model_function_large(input_dimension=input_dim,
                                                                  learning_rate=learning_rate))
    gs = GridSearchCV(estimator=kerasRegressor,
                      param_grid=param_grid,
                      cv=cv,
                      n_jobs=-1,
                      # scoring=scorer,
                      verbose=0)
    gs = gs.fit(x_train_cross_val,y_train_cross_val)
    print("gs best params",gs.best_params_)
    print("gs best score", gs.best_score_)

    return gs

epochs = 3500
batch_size = 80
learning_rate = 0.0015
input_dim = len(feature_cols)

param_grid = {
              'epochs':[1200,1500],
              'batch_size':[80],
              'learning_rate': [0.0015,0.002],
              'input_dimension':[input_dim]
              # 'optimizer':['Adam']
              # 'dropout_rate' : [0.0, 0.1, 0.2],
              # 'activation' :          ['relu', 'elu']
             }

do_manual_cv = False
fitted_model: Sequential
#temporary code
train_columns_length = len(feature_cols)
length = train_columns_length if train_columns_length > test_columns_length else test_columns_length

if do_manual_cv:
    with open(textfile_name, 'a') as f:
        print("bosss",textfile_name)
        ret = f.write("\n epochs " + str(epochs) + " batch_size " + str(batch_size) + " learning_rate " + str(learning_rate))
        print("return from write to file",ret)
    model = model_function_large(input_dimension=input_dim ,learning_rate=learning_rate ,instantiate=True)
    train_with_stopping_rounds = True
    if train_with_stopping_rounds:
        fitted_model = manual_cv_with_tfdataset(x_train_cross_val, y_train_cross_val, model,
                                                epochs=epochs, batch_size=batch_size, patience=250)
    else:
        fitted_model = manual_cv(x_train_cross_val, y_train_cross_val, model,
                                 epochs=epochs, batch_size=batch_size)

    y_prediction_for_holdout_set = fitted_model.predict(np.array(x_holdout_set))
else:
    gs = grid_cv(x_train_cross_val,y_train_cross_val,param_grid, learning_rate=learning_rate,cv=5)
    with open(textfile_name, 'a') as f:
        f.write("\n learning rate" + str(learning_rate))
        f.write("\n gs best params"+str(gs.best_params_))
        f.write("\n gs best score"+str(gs.best_score_))
    model = gs.estimator.build_fn()
    y_prediction_for_holdout_set = gs.predict(np.array(x_holdout_set))

# loss = fitted_model.evaluate(np.array(x_test), np.array(y_test))
# loss = mean_absolute_error(np.array(y_test),y_prediction_test)
loss = binary_crossentropy(np.array(y_holdout_set), y_prediction_for_holdout_set)
loss_on_holdoutset_text = ""
if do_manual_cv:
    loss_manual = loss.numpy().mean()
    loss_on_holdoutset_text = "manualcv loss on the hold out final test set:" + str(loss_manual)
else:
    loss_on_holdoutset_text = "gridsearchcv loss on the hold out final test set:" + str(loss)

print(loss_on_holdoutset_text)
with open(textfile_name,'a') as f:
    f.write("\n"+loss_on_holdoutset_text+"\n")
    model.summary(print_fn=lambda x: f.write(x + '\n'))


# prediction for sumbmission
if do_manual_cv:
    y_predict = fitted_model.predict(np.array(test))
else:
    y_predict = gs.predict(np.array(test))


def to_submit(pred_y, name_out):
    y_predict = list(itertools.islice(pred_y,test.shape[0]))
    y_predict = pd.DataFrame(minmax_scaler_y.inverse_transform(np.array(y_predict).
                                                               reshape(len(y_predict),1)),
                             columns=['SalePrice'])
    # y_predict = y_predict.join(ID)
    output = pd.DataFrame({'Id': test_original.index,
                           'SalePrice': y_predict.SalePrice})

    output.to_csv('csvs/'+name_out+'.csv', index=False)

to_submit(y_predict,"submission")

