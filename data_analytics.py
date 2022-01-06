import matplotlib.pyplot as plt
from pylab import rcParams
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Lot_Frontage_Filler import *

train_df = pd.read_csv('../data/train.csv', index_col='Id')
test_df = pd.read_csv('../data/test.csv', index_col='Id')

columns = train_df.columns.to_list()

num_columns = len(train_df.columns)
pd.set_option("display.max_columns", num_columns)
train_df.head(30)
print(train_df.describe())
print(test_df.describe())

print("fill train set")
fill_Lot_Frontage_Nans(train_df)
print("fill test set")
fill_Lot_Frontage_Nans(test_df)


# for column in columns:
#     print(column)
#     print(train_df[column].value_counts(dropna=False))

