import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Lot_Frontage_Filler import *
from pandas_profiling import ProfileReport
from features import *

from statsmodels.stats.outliers_influence import variance_inflation_factor


# train_df = pd.read_csv('../data/train.csv', index_col='Id')
# test_df = pd.read_csv('../data/test.csv', index_col='Id')
train_processed_file = "train_processed.csv"
test_processed_file = "test_processed.csv"
train_minmax_file = "train_minmax.csv"
test_minmax_file = "test_minmax.csv"
train_w_dummies_file = "train_proccesed_w_dummies.csv"
test_w_dummies_file = "test_processed_w_dummies.csv"


train_file = train_w_dummies_file
test_file = test_w_dummies_file
train_df = pd.read_csv(train_file)#, index_col='Id')
test_df = pd.read_csv(test_file)#, index_col='Id')


# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = train_df.columns
vif_data["VIF"] = [variance_inflation_factor(train_df.values, i)
                          for i in range(len(train_df.columns))]

# train_df = pd.read_csv('../data/train.csv', index_col='Id', keep_default_na=False)
# test_df = pd.read_csv('../data/test.csv', index_col='Id',keep_default_na=False)


# categorical_cols = get_categorical_features_from_df(train_df)
# train_df[categorical_cols] = train_df[categorical_cols].replace(np.nan,'NA')
# test_df[categorical_cols] = test_df[categorical_cols].replace(np.nan,'NA')

columns = train_df.columns.to_list()

profile_train = ProfileReport(train_df, title="Train Profiling Report",explorative=True)
profile_test = ProfileReport(test_df, title="Test Profiling Report",explorative=True)

profile_train.to_file(train_file + "_analysis.html")
profile_test.to_file(test_file+"_analysis.html")
print(profile_train)
print(profile_test)

# print("fill train set")
# fill_Lot_Frontage_Nans(train_df)
# print("fill test set")
# fill_Lot_Frontage_Nans(test_df)


# for column in columns:
#     print(column)
#     print(train_df[column].value_counts(dropna=False))

