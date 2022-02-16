import pandas as pd
from sklearn.compose import ColumnTransformer
from features import *
import numpy as np

def dataframe_feature_engineering_dummies(df:pd.DataFrame,preprocessor:ColumnTransformer,
                                          categorical_cols_definded_range,
                                          my_cols_with_saleprice,
                                          my_cols,
                                          is_train_data=True):
    df_arr = preprocessor.fit_transform(df)
    if is_train_data:
        new_df = pd.DataFrame(df_arr, columns=my_cols_with_saleprice)
    else:
        new_df = pd.DataFrame(df_arr, columns=my_cols)
    df_with_dummies = pd.get_dummies(new_df,columns=categorical_cols_definded_range, drop_first=True)


    remaining_indexes = df_with_dummies.columns.drop(list(df_with_dummies.filter(regex='inexistent|_NA')))

    df_with_dummies = df_with_dummies[remaining_indexes]

    return df_with_dummies

def dataframe_feature_engineering(df:pd.DataFrame, preprocessor:ColumnTransformer ,is_train_data=True):
    preprocessor.fit(df)
    pipe = preprocessor.transformers_[2]
    one_hot_encoder_pipe = pipe[1][-1:]
    cols_for_one_hot_encodeding = get_one_hot_encoded_cols()
    one_h_encoded_cols = one_hot_encoder_pipe.get_feature_names_out(cols_for_one_hot_encodeding)
    df_arr = preprocessor.transform(df)
    df_new_columns = df.columns
    if is_train_data:
        df_new_columns = df_new_columns.drop(cols_for_one_hot_encodeding, )
        df_new_columns = df_new_columns.drop(['SalePrice'])
        column_names = np.concatenate([df_new_columns, one_h_encoded_cols,y_list])
    else:
        df_new_columns = df_new_columns.drop(cols_for_one_hot_encodeding)
        column_names = np.concatenate([df_new_columns, one_h_encoded_cols])

    new_df = pd.DataFrame(df_arr, columns=column_names)
    return new_df

#TODO to be modiefied to encode categorical as ordinal
def dataframe_feature_engineering_ordinal_enc(df:pd.DataFrame, preprocessor:ColumnTransformer ,is_train_data=True):
    preprocessor.fit(df)
    pipe = preprocessor.transformers_[2]
    one_hot_encoder_pipe = pipe[1][-1:]
    cols_for_one_hot_encodeding = get_one_hot_encoded_cols()
    one_h_encoded_cols = one_hot_encoder_pipe.get_feature_names_out(cols_for_one_hot_encodeding)
    df_arr = preprocessor.transform(df)
    df_new_columns = df.columns
    if is_train_data:
        df_new_columns = df_new_columns.drop(cols_for_one_hot_encodeding, )
        df_new_columns = df_new_columns.drop(['SalePrice'])
        column_names = np.concatenate([df_new_columns, one_h_encoded_cols,y_list])
    else:
        df_new_columns = df_new_columns.drop(cols_for_one_hot_encodeding)
        column_names = np.concatenate([df_new_columns, one_h_encoded_cols])

    new_df = pd.DataFrame(df_arr, columns=column_names)
    return new_df