import pandas as pd
train_processed_file = "processed data/train_processed.csv"
test_processed_file = "processed data/test_processed.csv"
train_minmax_file = "processed data/train_minmax.csv"
test_minmax_file = "processed data/test_minmax.csv"
train_w_dummies_csv_file = "processed data/train_proccesed_w_dummies.csv"
test_w_dummies_csv_file = "processed data/test_processed_w_dummies.csv"

def load_dataFrame_from_csv(train_file, test_file):
    # train_df = pd.read_csv('../data/train.csv', index_col='Id')
    # test_df = pd.read_csv('../data/test.csv', index_col='Id')

    train_df = pd.read_csv(train_file)  # , index_col='Id')
    test_df = pd.read_csv(test_file)  # , index_col='Id')

    return (train_df,test_df)