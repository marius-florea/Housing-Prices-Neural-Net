import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Lot_Frontage_Filler import *
from pandas_profiling import ProfileReport
from features import *

train_df = pd.read_csv('../data/train.csv', index_col='Id', keep_default_na=False)
test_df = pd.read_csv('../data/test.csv', index_col='Id',keep_default_na=False)

# train_df = pd.read_csv('../data/train.csv', index_col='Id', keep_default_na=False)
# test_df = pd.read_csv('../data/test.csv', index_col='Id',keep_default_na=False)


# categorical_cols = get_categorical_features_from_df(train_df)
# train_df[categorical_cols] = train_df[categorical_cols].replace(np.nan,'NA')
# test_df[categorical_cols] = test_df[categorical_cols].replace(np.nan,'NA')

columns = train_df.columns.to_list()
columns_with_na_string = ['Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2',
                          'FireplaceQu','GarageType','GarageFinish','GarageQual','GarageCond','PoolArea',
                          'Fence','MiscFeature']
columns_with_true_nans = [col for col in columns if col not in columns_with_na_string]
train_df[columns_with_true_nans] = train_df[columns_with_true_nans].replace(to_replace = 'NA',value=np.nan)
columns_with_true_nans.remove('SalePrice')
test_df[columns_with_true_nans] = test_df[columns_with_true_nans].replace(to_replace = 'NA',value=np.nan)

num_columns = len(train_df.columns)
pd.set_option("display.max_columns", num_columns)
train_df.head(30)
profile_train = ProfileReport(train_df, title="Train Profiling Report",explorative=True)
profile_test = ProfileReport(test_df, title="Test Profiling Report",explorative=True)

profile_train.to_file("train_analysis.html")
profile_test.to_file("test_analysis.html")
print(profile_train)
print(profile_test)

# print("fill train set")
# fill_Lot_Frontage_Nans(train_df)
# print("fill test set")
# fill_Lot_Frontage_Nans(test_df)


# for column in columns:
#     print(column)
#     print(train_df[column].value_counts(dropna=False))

