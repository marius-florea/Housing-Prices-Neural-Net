from Lot_Frontage_Filler import *
import pandas as pd
from features import *
from processing_utils import *
from feature_engineering import *

def get_features_with_nr_of_nulls_larger_than(train_original,test_original,nrOfNulls: int):
    m = train_original.eq('NA').sum()
    nulls_in_train_df = train_original.eq('NA').sum()
    nulls_in_test_df = test_original.eq('NA').sum()
    train_labels = nulls_in_train_df[nulls_in_train_df > nrOfNulls].index.values
    test_labels = nulls_in_test_df[nulls_in_test_df > nrOfNulls].index.values
    intersection = np.intersect1d(test_labels,train_labels)
    return intersection

def differentiate_NA_and_nans_and_store_to_csv():
    # differentiating NA from nan code - move to a function
    train_original = pd.read_csv('../data/train.csv', index_col='Id', keep_default_na=False)
    test_original = pd.read_csv('../data/test.csv', index_col='Id', keep_default_na=False)
    columns_with_true_nans = get_columns_with_true_nans(train_original)
    train_original[columns_with_true_nans] = train_original[columns_with_true_nans].replace(to_replace='NA',value=np.nan)
    columns_with_true_nans.remove('SalePrice')
    test_original[columns_with_true_nans] = test_original[columns_with_true_nans].replace(to_replace='NA',value=np.nan)
    train_original.to_csv("processed data/train_processed.csv")
    test_original.to_csv("processed data/test_processed.csv")


def load_and_preprocess_dataframes(train_original, test_original):
    # columns_with_na_string = get_columns_with_true_nans()
    # train_original[columns_with_na_string] = train_original[columns_with_na_string].replace(value='NA', to_replace=np.nan)
    # test_original[columns_with_na_string] = test_original[columns_with_na_string].replace(value='NA', to_replace=np.nan)
    nrOfNulls = 1150
    features_with_many_nulls = get_features_with_nr_of_nulls_larger_than(train_original, test_original, nrOfNulls) #TODO with these features removed the score got worse!!!
    high_correlated_features = ['GarageCars','GarageYrBlt','TotRmsAbvGrd','TotalBsmtSF','BedroomAbvGr','BsmtFullBath']

    # labels_with_high_vif = ['BsmtFinSF2', 'TotalBsmtSF', '2ndFlrSF', 'YearBuilt',
    #                             'YearRemodAdd', ]  # 34 YrSold 57832.71
    # # labels_with_high_vif = labels_with_high_vif + ['YrSold', 'GarageYrBlt','Neighborhood']

     # features_to_remove = np.append(high_correlated_features,features_with_many_nulls)
    #TODO uncomment to drop features to be removed
    train_original.drop(labels=features_with_many_nulls, axis=1, inplace=True)
    test_original.drop(labels=features_with_many_nulls, axis=1, inplace=True)

    fill_Lot_Frontage_Nans(train_original)
    fill_Lot_Frontage_Nans(test_original)

    train_nunique = train_original.nunique()
    test_nunique = test_original.nunique()

    # test = test_original.select_dtypes(exclude=['object'])
    # test.fillna(0,inplace=True)

    y_list =['SalePrice']
    Y_train_original = train_original[y_list]
    # train_original.drop(['SalePrice'], axis=1, inplace=True)


    numerical_cols = get_numerical_features_from_df_with_margin(train_original)#get_high_corelated_numerical_features()
    categorical_cols_unique_range_4_15 = get_categorical_features_from_df_in_range(train_original,minValue=4,maxValue=15)#get_choosen_categorical_features()
    categorical_cols_unique_over_15 = get_categorical_features_from_df_above_upper_margin(train_original, maxValue=15)


    categorical_cols_matching_unique_count__range_4_15 = []
    categorical_cols_nonmatching_unique_count__range_4_15 = []
    for col in categorical_cols_unique_range_4_15:
        if train_original[col].nunique() == test_original[col].nunique():
            categorical_cols_matching_unique_count__range_4_15.append(col)
        else:
            # categorical_cols_matching_unique_count__range_4_15.append(col) #Temporary !! replace with line below
            categorical_cols_nonmatching_unique_count__range_4_15.append(col)

    my_cols = numerical_cols + categorical_cols_matching_unique_count__range_4_15 + \
              categorical_cols_nonmatching_unique_count__range_4_15 \
              + categorical_cols_unique_over_15

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


    # numerical_cols.remove('MSSubClass')#temporary move somewhere elese or not?
    # categorical_cols.remove('MSZoning')

    categorical_cols_unique_over_15_and_nonmatching_uniques= categorical_cols_unique_over_15 + categorical_cols_nonmatching_unique_count__range_4_15
    preprocessor_train = get_preprocessor(numerical_cols, categorical_cols_matching_unique_count__range_4_15,
                                          categorical_cols_unique_over_15_and_nonmatching_uniques, y_list)
    preprocessor_test = get_preprocessor(numerical_cols, categorical_cols_matching_unique_count__range_4_15,
                                         categorical_cols_unique_over_15_and_nonmatching_uniques)

    # processed_X_full = dataframe_feature_engineering(train_na_filled, preprocessor_train, categorical_cols_unique_range_4_15 )
    processed_X_full = dataframe_feature_engineering_dummies(train_na_filled, preprocessor_train,
                                                                 categorical_cols_matching_unique_count__range_4_15,
                                                             my_cols_with_saleprice=my_cols_with_saleprice,
                                                             my_cols=my_cols,
                                                             is_train_data=True)
    processed_X_test = dataframe_feature_engineering_dummies(test_na_filled, preprocessor_test,
                                                             categorical_cols_matching_unique_count__range_4_15,
                                                             my_cols_with_saleprice=my_cols_with_saleprice,
                                                             my_cols=my_cols,
                                                             is_train_data=False)

    # processed_X_full.drop(labels=labels_with_high_vif, axis=1, inplace=True)
    # processed_X_test.drop(labels=labels_with_high_vif, axis=1, inplace=True)


    return processed_X_full, processed_X_test