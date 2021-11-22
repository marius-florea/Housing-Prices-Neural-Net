
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder

def get_preprocessor(numerical_cols, categorical_cols, label_column=''):
    numerical_transformer = SimpleImputer(strategy='constant') # Your code here

    # Preprocessing for categorical data
    #
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('ordinal', OrdinalEncoder())#OneHotEncoder(handle_unknown='ignore'))
    ])

    numeric_categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant')),
        ('ordinal', OrdinalEncoder())
    ])

    # Bundle preprocessing for numerical and categorical data
    transformers_list = [
        ('passthrough-numeric', 'passthrough', numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
    if len(label_column)>0:
        transformers_list.append( ('passthrough-label', 'passthrough', label_column))

    preprocessor = ColumnTransformer(
        transformers=transformers_list, remainder="drop")
    return preprocessor