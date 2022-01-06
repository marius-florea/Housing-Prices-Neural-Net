import pandas as pd
import numpy as np

def find_neighbours(current_index, value, df, colname):
    df_current_index_removed = df[~df.index.isin([current_index])]
    exactmatch = df_current_index_removed[df_current_index_removed[colname] == value]
    # exactmatch = df[df[colname] == value and df.]#and df.loc[]
    # exactmatch.dro
    # exactmatch[~exactmatch['LotFrontage'].isna()]
    if not exactmatch.empty:
        index = (exactmatch.iloc[[0]]).index
        return index.values[0]
    else:
        df_lowerRows = df[df[colname] < value]
        df_upperRows = df[df[colname] > value]

        lower_diff = np.nan
        lowerneighbour_ind = np.nan

        if not df_lowerRows.empty:
            lowerneighbour_ind = df_lowerRows[colname].idxmax()
            lowerneighbour_value = df.loc[lowerneighbour_ind][colname]
            lower_diff = abs(value - lowerneighbour_value)

        upper_diff = np.nan
        upperneighbour_ind = np.nan

        if not df_upperRows.empty:
            upperneighbour_ind = df_upperRows[colname].idxmin()
            upperneighbour_value = df.loc[upperneighbour_ind][colname]
            upper_diff = abs(upperneighbour_value - value)

        if upper_diff != np.nan and lower_diff != np.nan:
            if lower_diff < upper_diff:
                return lowerneighbour_ind
            else:
                return upperneighbour_ind
        elif upper_diff != np.nan:
            return upperneighbour_ind
        elif lower_diff != np.nan:
            return lowerneighbour_ind
    return np.nan


def fill_Lot_Frontage_Nans(df: pd.DataFrame):
    lot_frontage_indexes_of_nans = df['LotFrontage'].index[df['LotFrontage'].apply(np.isnan)]
    train_df_removed_nan_lots = df.drop(lot_frontage_indexes_of_nans)
    nr_of_filled_lotfrontages = 0
    nr_of_not_filled_lotfrontages = 0
    nr_of_lots_to_fill = lot_frontage_indexes_of_nans.size

    for ind in lot_frontage_indexes_of_nans:
        lot_area = df.loc[ind]['LotArea']
        if not np.isnan(lot_area):
            similar_row_index = find_neighbours(ind,lot_area, df, 'LotArea')
            if not np.isnan(similar_row_index):
                df.at[ind,'LotFrontage'] = df.loc[similar_row_index]['LotFrontage']
                nr_of_filled_lotfrontages += 1
            else:
                nr_of_not_filled_lotfrontages+=1
                print(similar_row_index)

    nr_of_not_filled_lotfrontages = nr_of_lots_to_fill - nr_of_filled_lotfrontages
    print("nr_of_filled_lotfrontages nr_of_not_filled_lotfrontages nr_of_lots_to_fill",
          nr_of_filled_lotfrontages, nr_of_not_filled_lotfrontages, nr_of_lots_to_fill)
