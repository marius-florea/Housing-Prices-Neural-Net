import pandas as pd
from sklearn.metrics import mean_squared_error
import numpy as np

def remove_rows_with_nans(df:pd.DataFrame,nan_count=10):
    nan_values_per_row = {}
    for index,row in df.iterrows():
        nan_values_per_row[index] =  row.isna().sum()

    s = nan_values_per_row.items()
    index_val_item_pairs = nan_values_per_row.items()
    # sorted_index_nanscount = sorted(index_val_items, key=lambda x: x[1],reverse=True)
    indexes_with_more_than_x_nans = [k for k,v in index_val_item_pairs if v > nan_count]
    df = df.drop(indexes_with_more_than_x_nans)

    return df

def rmse(y_holdout_set,y_prediction_for_holdout_set):
    rmse = np.sqrt(mean_squared_error(y_holdout_set, y_prediction_for_holdout_set))
    return rmse