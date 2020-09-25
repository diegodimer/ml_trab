import pandas as pd

def normalize_df(self):
    """
    using the formula
    element - min   element / max element - min element
    """
    return (df-df.min()) / (df.max()-df.min())

def get_bootstrap(df, size, replacement=True):
    """
    df: PandasDataFrame, the dataframe to get the bootstrap from
    size: size of the bootstrap
    replacement: Bool, if bootstrap should use replacement, if
        set to true, the same entry could be in the bootstrap 
        more than once, default to true
    returns a PandasDataFrame with the bootstrap
    """
    return df.sample(size, replace=replacement)

def get_test_set_from_bootstrap(df, bootstrap):
    """
    gets the test set from the df, the rows in the test set are the ones
    that are not in the bootstrap (its the difference between these two sets)
    df: PandasDataFrame, the original dataframe
    bootstrap: PandasDataFrame, the bootstrap generated
    returns the difference between df and bootstrap
    """
    return pd.concat([df, bootstrap]).drop_duplicates(keep=False)
