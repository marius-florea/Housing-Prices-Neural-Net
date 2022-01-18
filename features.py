
def get_high_corelated_numerical_features():
    cols = [#'OverallQual',
    'GrLivArea','GarageCars','GarageArea','TotalBsmtSF',
    'FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd',
    'GarageYrBlt','MasVnrArea','Fireplaces','BsmtFinSF1','LotFrontage',
    'WoodDeckSF','OpenPorchSF','HalfBath',
    'LotArea','BsmtFullBath',
    # 'BsmtUnfSF',
    'BedroomAbvGr',
    'ScreenPorch','PoolArea','MoSold','3SsnPorch',#'Property_Quality', uncommented for now !!!! TODO
    # 'FloorsTotalSquareFeet',
    '1stFlrSF','2ndFlrSF'
    ]
    return cols

def get_high_corelated_categorical_features():
    cols = ['Street','KitchenQual','GarageCond','SaleType','BsmtQual','CentralAir','PavedDrive','RoofMatl','MiscFeature',
            'MSSubClass','LotConfig','BsmtExposure','BldgType',
            #'OverallCond',#'HeatingQC',#'Functional',#'MiscVal','BsmtCond'
    ]
    return cols
#MiscVal cat or num


def get_choosen_numerical_features():
    cols = [#'OverallQual',
            'GarageCars','GrLivArea','TotalBsmtSF','BsmtFinSF1','YearRemodAdd',
                   'TotRmsAbvGrd','Fireplaces','YearBuilt','LotArea', 'MoSold','1stFlrSF','2ndFlrSF', 'MasVnrArea',
                   # '****',
                    'LotFrontage', 'FullBath','BedroomAbvGr', 'KitchenAbvGr', 'PoolArea',
                    # remainder
                    'HalfBath', 'GarageYrBlt','GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'YrSold',
                   ]
    return cols

def get_choosen_categorical_features():
    cols = ['CentralAir','KitchenQual',
                        'SaleType',
                        'RoofStyle',
                        #'*****',
                        'Condition1','Foundation','HeatingQC','Electrical','LotShape','LandContour',
                        'BsmtQual', 'GarageFinish','GarageType','BsmtFinType1','BsmtExposure',
                        'GarageQual',
                        'MSZoning',
                        # # remainder
                        'LandSlope','Heating','HouseStyle','ExterQual',
                        'Street', 'Utilities', 'LotConfig','Neighborhood','Condition2','BldgType',
                         'RoofMatl', 'Exterior1st', 'ExterCond','Exterior2nd','MasVnrType',
                        'BsmtCond', #was better than quality
                        'BsmtFinType2',# kept score the same
                        'Functional', 'GarageCond','PavedDrive', 'MSSubClass','SaleCondition',
                        #'PoolQC', 'MiscFeature',
                        # 'Alley','Fence'
                     ]
    return cols

import pandas as pd
#4 best margin till now

def get_columns_with_na_string():
    return ['Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                              'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolArea',
                              'Fence', 'MiscFeature','PoolQC']

def get_columns_with_true_nans(df:pd.DataFrame):
    columns = df.columns.to_list()
    columns_with_na_string = get_columns_with_na_string()
    columns_with_true_nans = [col for col in columns if col not in columns_with_na_string]
    return columns_with_true_nans

def get_numerical_features_from_df(df:pd.DataFrame):
    # Select numerical columns
    numerical_cols = [cname for cname in df.columns if
                      df[cname].dtype in ['int64', 'float64']
                      ]
    numerical_cols.remove('SalePrice')
    return numerical_cols

def get_categorical_features_from_df(df:pd.DataFrame):
    # Select numerical columns
    categorical_cols = [cname for cname in df.columns if
                      df[cname].dtype in ['object']
                      ]
    return categorical_cols

unique_margin = 4
def get_numerical_features_from_df_with_margin(df:pd.DataFrame):
    # Select numerical columns
    numerical_cols = [cname for cname in df.columns if
                      (df[cname].dtype in ['int64', 'float64']
                       and df[cname].nunique() >= unique_margin)
                      ]
    numerical_cols.remove('SalePrice')
    return numerical_cols

def get_categorical_features_from_df_with_margin(df:pd.DataFrame):
    # "Cardinality" means the number of unique values in a column
    # Select categorical co lumns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in df.columns if
                        (df[cname].nunique() >= (unique_margin) and
                         df[cname].dtype == "object")]
    return categorical_cols

def get_categorical_features_from_df_in_range(df:pd.DataFrame,minValue:float=4,maxValue:float=15):
    # "Cardinality" means the number of unique values in a column
    # Select categorical co lumns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in df.columns if
                        (df[cname].nunique() >= (minValue) and df[cname].nunique()<=maxValue
                         and df[cname].dtype == "object")]
    return categorical_cols

def get_categorical_features_from_df_above_upper_margin(df:pd.DataFrame, maxValue:float=15):
    # "Cardinality" means the number of unique values in a column
    # Select categorical co lumns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [cname for cname in df.columns if
                        (df[cname].nunique() > maxValue
                         and df[cname].dtype == "object")]
    return categorical_cols

def get_one_hot_encoded_cols():
    return ['MSSubClass','MSZoning']




