
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