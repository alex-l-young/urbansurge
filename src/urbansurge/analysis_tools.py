##########################################################################
# Fault detection analysis tools.
# Alex Young
##########################################################################

# Library imports.
import pandas as pd
import numpy as np


def flatten_df(df):
    '''
    Flattens Pandas data frame columns to rows.
    :param df:
    :return:
    '''

    df_T = df.T

    single_row_dict = {}
    for i, name in enumerate(df_T.index):
        col_dict = {f'{name}_{j}': df_T.iloc[i, j] for j in range(df_T.shape[1])}

        # Append to single_row_dict.
        single_row_dict.update(col_dict)

    single_row_df = pd.DataFrame(single_row_dict, index=[0])

    return single_row_df


if __name__ == '__main__':
    data = {'dt': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
          'node1': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115],
          'node2': [201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215],
          'node3': [301, 302, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 313, 314, 315]}
    df = pd.DataFrame(data)
    flatten_df(df)