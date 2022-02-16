import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Lot_Frontage_Filler import *
from pandas_profiling import ProfileReport
from features import *
from data_analytics_utils import *
from statsmodels.stats.outliers_influence import variance_inflation_factor


(train_df, test_df) = load_dataFrame_from_csv(train_w_dummies_csv_file, test_w_dummies_csv_file)

# VIF dataframe
vif_data = pd.DataFrame()
vif_data["feature"] = train_df.columns
vif_data["VIF"] = [variance_inflation_factor(train_df.values, i)
                          for i in range(len(train_df.columns))]

selection_vif = vif_data[vif_data["VIF"] > 100]
selection_vif = selection_vif.sort_values(by='VIF',ascending=False)
# train_df = pd.read_csv('../data/train.csv', index_col='Id', keep_default_na=False)
# test_df = pd.read_csv('../data/test.csv', index_col='Id',keep_default_na=False)
train_df_reduced = train_df.copy()
#'BsmtUnfSF'
labels_with_high_vif = ['BsmtFinSF2', 'TotalBsmtSF','2ndFlrSF', 'YearBuilt', 'YearRemodAdd', ]  #34 YrSold 57832.71
labels_with_high_vif = labels_with_high_vif + ['YrSold','GarageYrBlt''Neighborhood']
#MSZoning_RL RoofStyle_Gable ExterCond_TA Functional_Typ GarageType_Attchd GarageCond_TA

train_df_reduced.drop(labels=labels_with_high_vif, axis=1, inplace=True)
vif_on_reduced_dataFrame = pd.DataFrame()
vif_on_reduced_dataFrame["feature"] = train_df_reduced.columns
vif_on_reduced_dataFrame["VIF"] = [variance_inflation_factor(train_df_reduced.values, i)
                          for i in range(len(train_df_reduced.columns))]#TODO check how this syntax works
selection_reduced_vif = vif_on_reduced_dataFrame[vif_on_reduced_dataFrame["VIF"] > 100]
selection_reduced_vif = selection_reduced_vif.sort_values(by='VIF',ascending=False)

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

