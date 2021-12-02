import pandas as pd
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