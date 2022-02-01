import numpy as np
import pandas as pd

train_processed_file = "train_processed.csv"
test_processed_file = "test_processed.csv"
# train_minmax_file = "train_minmax.csv"
# test_minmax_file = "test_minmax.csv"

train_file = train_processed_file
test_file = test_processed_file
train_df = pd.read_csv(train_file)#, index_col='Id')
test_df = pd.read_csv(test_file)#, index_col='Id')

nulls_in_train_df = train_df.isnull().sum()
print(nulls_in_train_df)
nulls_in_test_df = test_df.isnull().sum()
print(nulls_in_test_df)

corr_matrix = train_df.corr().abs()

#the matrix is symmetric so we need to extract upper triangle matrix without diagonal (k = 1)
sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
                  .stack()
                  .sort_values(ascending=False))

print("Top Absolute Correlations")
print(sol[:20])
print()


# Create correlation matrix
# corr_matrix = train_df.corr().abs()
#
# # Select upper triangle of correlation matrix
# upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
#
# # Find features with correlation greater than 0.95
# to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
#
# # Drop features
# train_df.drop(to_drop, axis=1, inplace=True)
# print(train_df.columns)