
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from features import *

def get_preprocessor(numerical_cols, categorical_cols_definded_range, categorical_cols_over_15_and_non_matching, label_column=''):
    numerical_transformer = SimpleImputer(strategy='constant') # Your code here
    # Preprocessing for categorical data
    #

    # numeric_categorical_transformer = Pipeline(steps=[
    #     # ('imputer', SimpleImputer(strategy='constant')),
    #     ('ordinal', OrdinalEncoder())
    # ])
    categorical_transformer_over_15 = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('ordinal', OrdinalEncoder())
    ])

    categorical_transformer_definded_range = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
    ])

    # one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
    # numeric_transformer_one_hot = Pipeline(steps=[
    #     ('imputer', SimpleImputer(strategy='constant')),
    #     ('one_hot', one_hot_encoder)
    # ])

    # Bundle preprocessing for numerical and categorical data
    cols_for_one_hot_encoding = get_one_hot_encoded_cols()
    transformers_list = [
        ('passthrough-numeric', 'passthrough', numerical_cols),
        ('categorical_transformer_definded_range', categorical_transformer_definded_range, categorical_cols_definded_range),
        ('categorical_transformer_over_15', categorical_transformer_over_15, categorical_cols_over_15_and_non_matching),
        # ('cat2',numeric_transformer_one_hot,cols_for_one_hot_encoding)
    ]
    if len(label_column) > 0:
        transformers_list.append( ('passthrough-label', 'passthrough', label_column))

    preprocessor = ColumnTransformer(
        transformers=transformers_list, remainder="drop")
    return preprocessor

